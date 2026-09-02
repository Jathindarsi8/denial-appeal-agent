"""
Day 9: checkpoint and resume.

A run that dies partway through currently loses everything. That has cost real
money on a free tier twice this week. On 27 Aug a run made two successful tool
calls and then hit the rate limit, and all of it was thrown away. On 1 Sep an
upstream fault burned five requests out of a twenty-per-day budget and returned
nothing, which then made the next two runs fail on quota.

Every one of those discarded steps was a request that had already been paid
for. Redoing completed work is the most expensive thing this system does.

So: write the state after every step, and let a run pick up where it stopped.

Three things this has to get right.

*A checkpoint must not outlive the code that made it.* If the system prompt,
the guardrails or the claim have changed since the checkpoint was written,
resuming would silently mix two different versions of the agent in one run.
Each checkpoint carries a fingerprint of the inputs and refuses to resume
against a mismatch.

*Resuming is not the same as re-running.* The conversation is rebuilt from
saved messages, so the model sees the same history — but days 4, 5 and 8 all
showed the multi-turn path is not deterministic even at temperature 0. A
resumed run may reach a different answer than an uninterrupted one would have.
That is a property of the agent, not of this file, and it is recorded on the
run so nobody later mistakes a resumed result for a clean one.

*A finished run must not be resumable.* Once the guardrails have decided, the
checkpoint is closed. Otherwise a rerun could reopen a settled case.

    python checkpoint.py list          what is currently resumable
    python checkpoint.py clear <key>   drop one
    python checkpoint.py sweep         drop everything already completed
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from store import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_key      TEXT PRIMARY KEY,
    claim_id     TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    steps_taken  INTEGER NOT NULL,
    state_json   TEXT NOT NULL,
    complete     INTEGER NOT NULL DEFAULT 0,
    resumed      INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ckpt_claim ON checkpoints(claim_id);
"""


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def fingerprint(denial, system_prompt: str, model: str) -> str:
    """What a resume is only valid against. Change any of it and the saved
    conversation belongs to a different agent than the one now running."""
    from agent import CONFIDENCE_FLOOR, NEVER_AUTO_APPEAL

    parts = [
        denial.claim_id,
        denial.carc,
        denial.payer_explanation,
        denial.documentation_summary,
        system_prompt,
        model,
        str(CONFIDENCE_FLOOR),
        ",".join(sorted(NEVER_AUTO_APPEAL)),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]


def run_key(denial, model: str) -> str:
    """One resumable run per claim per model. A second attempt at the same
    claim resumes the first rather than starting beside it."""
    return f"{denial.claim_id}:{model}"


# ------------------------------------------------------------------ writing

def _serialise(state) -> str:
    return json.dumps({
        "steps_taken": state.steps_taken,
        "code_category": state.code_category,
        "code_meaning": state.code_meaning,
        "messages": state.messages,
        "tools_called": state.tools_called,
        "observations": state.observations,
        "trace": state.trace,
        "judgment": state.judgment.model_dump() if state.judgment else None,
    })


def save(state, key: str, fp: str, complete: bool = False) -> None:
    init()
    with connect() as conn:
        conn.execute(
            """INSERT INTO checkpoints
               (run_key, claim_id, fingerprint, steps_taken, state_json,
                complete, resumed, updated_at)
               VALUES (?,?,?,?,?,?,
                       COALESCE((SELECT resumed FROM checkpoints
                                 WHERE run_key = ?), 0), ?)
               ON CONFLICT(run_key) DO UPDATE SET
                 fingerprint = excluded.fingerprint,
                 steps_taken = excluded.steps_taken,
                 state_json  = excluded.state_json,
                 complete    = excluded.complete,
                 updated_at  = excluded.updated_at""",
            (key, state.denial.claim_id, fp, state.steps_taken,
             _serialise(state), int(complete), key,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()


def mark_resumed(key: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE checkpoints SET resumed = 1 WHERE run_key = ?",
                     (key,))
        conn.commit()


# ------------------------------------------------------------------ reading

def load(key: str, fp: str) -> Optional[dict]:
    """Returns the saved state, or None with a printed reason. Refusing to
    resume is normal and safe; resuming against changed code is not."""
    init()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM checkpoints WHERE run_key = ?", (key,)).fetchone()

    if row is None:
        return None
    if row["complete"]:
        return None
    if row["fingerprint"] != fp:
        print(f"  checkpoint for {key} was written against different inputs "
              f"or code, starting fresh")
        return None
    if row["steps_taken"] < 1:
        return None

    return json.loads(row["state_json"])


def restore(state, saved: dict) -> None:
    """Put the saved work back on a fresh state object."""
    from agent import ModelAction

    state.steps_taken = saved["steps_taken"]
    state.code_category = saved["code_category"]
    state.code_meaning = saved["code_meaning"]
    state.messages = saved["messages"]
    state.tools_called = saved["tools_called"]
    state.observations = saved["observations"]
    state.trace = saved["trace"]
    if saved.get("judgment"):
        state.judgment = ModelAction.model_validate(saved["judgment"])

    state.log(f"resumed from checkpoint at step {state.steps_taken}, "
              f"{len(state.tools_called)} tool call(s) already paid for")


# ------------------------------------------------------------------ cli

def list_open() -> None:
    init()
    with connect() as conn:
        rows = conn.execute(
            """SELECT run_key, claim_id, steps_taken, complete, resumed,
                      updated_at
               FROM checkpoints ORDER BY updated_at DESC""").fetchall()

    if not rows:
        print("no checkpoints")
        return

    open_n = sum(1 for r in rows if not r["complete"])
    print(f"{len(rows)} checkpoint(s), {open_n} resumable\n")
    for r in rows:
        status = "done" if r["complete"] else f"OPEN at step {r['steps_taken']}"
        flag = "  (was resumed)" if r["resumed"] else ""
        print(f"  {r['run_key']:<34} {status}{flag}")
        print(f"    {r['updated_at']}")


def clear(key: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM checkpoints WHERE run_key = ?", (key,))
        conn.commit()
    print(f"cleared {key}")


def sweep() -> None:
    with connect() as conn:
        n = conn.execute("DELETE FROM checkpoints WHERE complete = 1").rowcount
        conn.commit()
    print(f"removed {n} completed checkpoint(s)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        list_open()
    elif cmd == "clear" and len(sys.argv) > 2:
        clear(sys.argv[2])
    elif cmd == "sweep":
        sweep()
    else:
        print(__doc__)
