"""
Day 7: memory, backed by SQLite.

Two problems this closes.

The claims were hardcoded Python objects inside the scripts. A workflow reads
its work from somewhere; it doesn't have five examples typed into the source.

And the agent had no memory at all. Every run started from nothing, which is
fine when a run is deterministic and dangerous when it isn't. Day 4 and 5
showed the same claim can come back escalate on one run and appeal on the next.
An agent that can't see what happened last time will happily contradict a human
who already made a decision.

Memory here is a lookup, not a vector store. The useful questions are exact:
has this claim been seen, has this member been through this before, what
normally happens with this denial code. Those are joins, not similarity search,
and pretending otherwise would be architecture for its own sake.

Three tables:

  claims       the work queue
  runs         every agent run, replacing runs.jsonl
  resolutions  what a human actually decided, which is the only ground truth
               in the system

    python store.py init         create the database
    python store.py migrate      pull runs.jsonl into runs
    python store.py stats        what the log says so far
    python store.py history CLM-100046
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path("denials.db")
LEGACY_LOG = Path("runs") / "runs.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id            TEXT PRIMARY KEY,
    patient_id          TEXT NOT NULL,
    payer               TEXT NOT NULL,
    amount              REAL NOT NULL,
    carc                TEXT NOT NULL,
    rarc                TEXT,
    payer_explanation   TEXT NOT NULL,
    documentation_summary TEXT NOT NULL,
    received_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL,
    ts                  TEXT NOT NULL,
    model               TEXT NOT NULL,
    code_category       TEXT,
    steps_taken         INTEGER,
    tools_called        TEXT,          -- json array
    evidence_retrieved  INTEGER,       -- 0/1
    proposed_decision   TEXT,
    confidence          REAL,
    final_decision      TEXT,
    stop_reason         TEXT,
    guardrail_overrode  INTEGER,       -- 0/1
    appeal_drafted      INTEGER        -- 0/1
);

CREATE INDEX IF NOT EXISTS idx_runs_claim ON runs(claim_id);
CREATE INDEX IF NOT EXISTS idx_runs_stop  ON runs(stop_reason);

-- What a human decided. The agent proposes and the guardrails authorize, but
-- neither of those is ground truth. This table is.
CREATE TABLE IF NOT EXISTS resolutions (
    claim_id     TEXT PRIMARY KEY,
    decided_by   TEXT NOT NULL,
    decision     TEXT NOT NULL,        -- appeal / do_not_appeal / corrected_claim
    rationale    TEXT,
    decided_at   TEXT NOT NULL
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init(path: Path = DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
    print(f"initialised {path}")


# ------------------------------------------------------------------- writing

def upsert_claim(claim, conn: Optional[sqlite3.Connection] = None) -> None:
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute(
            """INSERT INTO claims
               (claim_id, patient_id, payer, amount, carc, rarc,
                payer_explanation, documentation_summary, received_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(claim_id) DO UPDATE SET
                 payer_explanation = excluded.payer_explanation,
                 documentation_summary = excluded.documentation_summary""",
            (claim.claim_id, claim.patient_id, claim.payer, claim.amount,
             claim.carc, claim.rarc, claim.payer_explanation,
             claim.documentation_summary,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()
    finally:
        if own:
            conn.close()


def record_run(record: dict, conn: Optional[sqlite3.Connection] = None) -> None:
    """Takes the same dict audit.build_record produces, so both can run during
    the transition off JSONL."""
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, claim_id, ts, model, code_category, steps_taken,
                tools_called, evidence_retrieved, proposed_decision, confidence,
                final_decision, stop_reason, guardrail_overrode, appeal_drafted)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["run_id"], record["claim_id"], record["timestamp"],
                record["model"], record.get("code_category"),
                record.get("steps_taken"),
                json.dumps(record.get("tools_called", [])),
                int(bool(record.get("evidence_retrieved"))),
                record.get("proposed_decision"), record.get("confidence"),
                record.get("final_decision"), record.get("stop_reason"),
                int(bool(record.get("guardrail_overrode_model"))),
                int(bool(record.get("appeal_drafted"))),
            ),
        )
        conn.commit()
    finally:
        if own:
            conn.close()


def record_resolution(claim_id: str, decided_by: str, decision: str,
                      rationale: str = "") -> None:
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO resolutions
               (claim_id, decided_by, decision, rationale, decided_at)
               VALUES (?,?,?,?,?)""",
            (claim_id, decided_by, decision, rationale,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()


# ------------------------------------------------------------------- memory

@dataclass
class ClaimMemory:
    """What the agent is allowed to know about a claim before it starts."""
    seen_before: int = 0
    prior_decisions: list[str] = None
    prior_confidences: list[float] = None
    human_resolution: Optional[str] = None
    human_rationale: Optional[str] = None
    member_prior_claims: int = 0
    category_history: Optional[str] = None

    def is_empty(self) -> bool:
        return (self.seen_before == 0
                and self.human_resolution is None
                and self.member_prior_claims == 0
                and not self.category_history)

    def as_context(self) -> str:
        """Rendered into the model's prompt. Facts only. No suggestion about
        what to do with them, because a model told 'this was escalated before'
        will otherwise take that as an instruction to escalate again."""
        if self.is_empty():
            return "No prior history for this claim, member, or denial code."

        lines = []
        if self.human_resolution:
            lines.append(
                f"A human already resolved this claim: {self.human_resolution}."
                + (f" Reason given: {self.human_rationale}"
                   if self.human_rationale else "")
            )
        if self.seen_before:
            decisions = ", ".join(self.prior_decisions or [])
            lines.append(
                f"This claim has been processed {self.seen_before} time(s) "
                f"before. Previous outcomes: {decisions}."
            )
        if self.member_prior_claims:
            lines.append(
                f"This member has {self.member_prior_claims} other denied "
                f"claim(s) on file."
            )
        if self.category_history:
            lines.append(f"For this denial code across all claims: {self.category_history}")
        return "\n".join(lines)


def recall(claim_id: str, patient_id: str, carc: str,
           path: Path = DB_PATH) -> ClaimMemory:
    if not path.exists():
        return ClaimMemory(prior_decisions=[], prior_confidences=[])

    with connect(path) as conn:
        mem = ClaimMemory(prior_decisions=[], prior_confidences=[])

        rows = conn.execute(
            """SELECT final_decision, confidence FROM runs
               WHERE claim_id = ? ORDER BY ts""",
            (claim_id,),
        ).fetchall()
        mem.seen_before = len(rows)
        mem.prior_decisions = [r["final_decision"] for r in rows if r["final_decision"]]
        mem.prior_confidences = [r["confidence"] for r in rows if r["confidence"] is not None]

        res = conn.execute(
            "SELECT decision, rationale FROM resolutions WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if res:
            mem.human_resolution = res["decision"]
            mem.human_rationale = res["rationale"]

        row = conn.execute(
            "SELECT COUNT(*) AS n FROM claims WHERE patient_id = ? AND claim_id != ?",
            (patient_id, claim_id),
        ).fetchone()
        mem.member_prior_claims = row["n"] if row else 0

        cat = conn.execute(
            """SELECT r.final_decision, COUNT(*) AS n
               FROM runs r JOIN claims c ON c.claim_id = r.claim_id
               WHERE c.carc = ? AND r.claim_id != ?
               GROUP BY r.final_decision ORDER BY n DESC""",
            (carc, claim_id),
        ).fetchall()
        if cat:
            mem.category_history = ", ".join(
                f"{r['final_decision']} {r['n']}x" for r in cat if r["final_decision"]
            )

    return mem


# ------------------------------------------------------------------- reading

def migrate(log: Path = LEGACY_LOG, path: Path = DB_PATH) -> None:
    if not log.exists():
        print(f"{log} not found, nothing to migrate")
        return
    init(path)
    n = 0
    with connect(path) as conn:
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            conn.execute(
                """INSERT OR IGNORE INTO claims
                   (claim_id, patient_id, payer, amount, carc, rarc,
                    payer_explanation, documentation_summary, received_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (rec["claim_id"], rec.get("patient_id", "unknown"),
                 rec.get("payer", "unknown"), rec.get("amount", 0.0),
                 rec.get("carc", ""), rec.get("rarc"),
                 "", "", rec["timestamp"]),
            )
            record_run(rec, conn)
            n += 1
        conn.commit()
    print(f"migrated {n} runs from {log}")


def stats(path: Path = DB_PATH) -> None:
    if not path.exists():
        print(f"{path} not found. Run: python store.py init")
        return
    with connect(path) as conn:
        c = conn.execute("SELECT COUNT(*) n FROM claims").fetchone()["n"]
        r = conn.execute("SELECT COUNT(*) n FROM runs").fetchone()["n"]
        res = conn.execute("SELECT COUNT(*) n FROM resolutions").fetchone()["n"]
        print(f"claims {c}   runs {r}   human resolutions {res}\n")

        print("what decided each run:")
        for row in conn.execute(
            """SELECT stop_reason, COUNT(*) n FROM runs
               GROUP BY stop_reason ORDER BY n DESC"""):
            print(f"  {row['n']:>3}  {row['stop_reason']}")

        print("\nclaims run more than once:")
        rows = conn.execute(
            """SELECT claim_id, COUNT(*) n,
                      COUNT(DISTINCT final_decision) distinct_outcomes
               FROM runs GROUP BY claim_id HAVING n > 1
               ORDER BY distinct_outcomes DESC, n DESC""").fetchall()
        if not rows:
            print("  none")
        for row in rows:
            flag = "  UNSTABLE" if row["distinct_outcomes"] > 1 else ""
            print(f"  {row['claim_id']}  {row['n']} runs, "
                  f"{row['distinct_outcomes']} distinct outcome(s){flag}")

        floor = conn.execute(
            """SELECT COUNT(*) n FROM runs
               WHERE stop_reason LIKE 'confidence_below_floor%'""").fetchone()["n"]
        print(f"\nconfidence floor fired on {floor} of {r} runs")


def history(claim_id: str, path: Path = DB_PATH) -> None:
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT ts, model, proposed_decision, confidence,
                      final_decision, stop_reason, tools_called
               FROM runs WHERE claim_id = ? ORDER BY ts""",
            (claim_id,)).fetchall()
        if not rows:
            print(f"no runs recorded for {claim_id}")
            return
        print(f"{claim_id}  ({len(rows)} runs)\n")
        for r in rows:
            tools = ", ".join(json.loads(r["tools_called"] or "[]")) or "none"
            conf = r["confidence"]
            print(f"  {r['ts']}  {r['model']}")
            print(f"    proposed {r['proposed_decision']} "
                  f"at {conf if conf is None else f'{conf:.2f}'} "
                  f"-> final {r['final_decision']}")
            print(f"    {r['stop_reason']}   tools: {tools}")

        res = conn.execute(
            "SELECT * FROM resolutions WHERE claim_id = ?", (claim_id,)).fetchone()
        if res:
            print(f"\n  HUMAN DECISION: {res['decision']} by {res['decided_by']}")
            if res["rationale"]:
                print(f"  {res['rationale']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "init":
        init()
    elif cmd == "migrate":
        migrate()
    elif cmd == "stats":
        stats()
    elif cmd == "history":
        if len(sys.argv) < 3:
            print("usage: python store.py history <claim_id>")
        else:
            history(sys.argv[2])
    else:
        print(__doc__)
