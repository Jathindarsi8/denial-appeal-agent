"""
Day 5: the audit trail.

Until now a run left nothing behind. The trace printed to the terminal and
disappeared. That was survivable while the agent was deterministic. Day 4
showed it isn't: the same claim came back escalate three times and appeal
twice, same model, temperature 0.

That breaks two things at once.

A human picking up an escalation needs to know *which* run they're looking at
— what it retrieved, what it proposed, and which specific rule stopped it.
"The agent escalated" is not an answer when the agent might have said something
different thirty seconds earlier.

And week 3 can't compute a pass rate over runs that were never persisted.

So: one JSON object per run, appended to a JSONL file. Append-only, one line
per run, no rewriting history. Every field is either copied from the run or
derived from it — nothing is inferred after the fact.

    python audit.py                  # summarise runs/runs.jsonl
    python audit.py runs/other.jsonl # summarise a different log
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent import AgentState, CONFIDENCE_FLOOR

DEFAULT_LOG = Path("runs") / "runs.jsonl"


# ------------------------------------------------------------------ writing

def build_record(state: AgentState, model: str) -> dict:
    """Flatten a finished run into one auditable object."""
    d = state.denial
    j = state.judgment

    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,

        # what came in
        "claim_id": d.claim_id,
        "patient_id": d.patient_id,
        "payer": d.payer,
        "amount": d.amount,
        "carc": d.carc,
        "rarc": d.rarc,

        # what the deterministic layer knew before the model ran
        "code_category": state.code_category,
        "code_meaning": state.code_meaning,

        # what the agent did
        "steps_taken": state.steps_taken,
        "tools_called": list(state.tools_called),
        "evidence_retrieved": bool(state.observations),
        "observations": list(state.observations),

        # what the model proposed
        "model_category": j.denial_category if j else None,
        "model_root_cause": j.root_cause if j else None,
        "proposed_decision": j.proposed_decision if j else None,
        "confidence": j.confidence if j else None,
        "model_reasoning": j.reasoning_summary if j else None,

        # what was actually authorized, and by which rule
        "final_decision": state.decision.value if state.decision else None,
        "stop_reason": state.stop_reason,
        "guardrail_overrode_model": _was_overridden(state),
        "appeal_drafted": state.appeal_draft is not None,

        "confidence_floor": CONFIDENCE_FLOOR,
        "trace": list(state.trace),
    }


def _was_overridden(state: AgentState) -> bool:
    """True when the final decision differs from what the model proposed."""
    if state.judgment is None or state.decision is None:
        return False
    return state.judgment.proposed_decision != state.decision.value


def record_run(state: AgentState, model: str,
               path: Optional[Path | str] = None) -> dict:
    """Append one run to the log. Returns the record it wrote."""
    target = Path(path) if path else DEFAULT_LOG
    target.parent.mkdir(parents=True, exist_ok=True)

    record = build_record(state, model)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


# ------------------------------------------------------------------ reading

def load(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(path: Path | str = DEFAULT_LOG) -> None:
    records = load(path)
    if not records:
        print(f"no runs recorded in {path}")
        return

    print(f"log:    {path}")
    print(f"runs:   {len(records)}")
    print(f"claims: {len(set(r['claim_id'] for r in records))}\n")

    by_claim: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_claim[r["claim_id"]].append(r)

    unstable = []

    for claim_id, runs in sorted(by_claim.items()):
        finals = Counter(r["final_decision"] for r in runs)
        proposed = Counter(r["proposed_decision"] for r in runs if r["proposed_decision"])
        confidences = [r["confidence"] for r in runs if r["confidence"] is not None]
        models = sorted(set(r["model"] for r in runs))

        print(f"{claim_id}  ({len(runs)} run{'s' if len(runs) > 1 else ''}, "
              f"{', '.join(models)})")
        if proposed:
            print(f"  proposed   {_fmt(proposed, len(runs))}")
        print(f"  final      {_fmt(finals, len(runs))}")      
       

        if confidences:
            line = f"  confidence {min(confidences):.2f}–{max(confidences):.2f}"
            if len(confidences) > 1:
                line += f"   stdev {statistics.stdev(confidences):.3f}"
            print(line)

        overrides = sum(1 for r in runs if r["guardrail_overrode_model"])
        if overrides:
            print(f"  guardrail overrode the model on {overrides}/{len(runs)}")

        # Only meaningful for runs where the model actually judged. A run that
        # escalated on an unmapped code never reached the model at all, and
        # counting it here would misreport why it stopped.
        judged = [r for r in runs if r["proposed_decision"] is not None]
        no_evidence = sum(1 for r in judged if not r["evidence_retrieved"])
        if no_evidence:
            print(f"  judged with nothing retrieved on {no_evidence}/{len(judged)}")

        reasons = Counter(r["stop_reason"] for r in runs)
        for reason, n in reasons.most_common():
            print(f"    {n}x  {reason}")

        if len(proposed) > 1:
            unstable.append(claim_id)
            print("  UNSTABLE: the model did not propose the same thing every time")

        print()

    print("=" * 68)
    floor_fired = sum(
        1 for r in records
        if r["stop_reason"] and r["stop_reason"].startswith("confidence_below_floor")
    )
    print(f"confidence floor fired on {floor_fired}/{len(records)} runs")

    all_reasons = Counter(r["stop_reason"] for r in records)
    print("\nwhat actually decided each run:")
    for reason, n in all_reasons.most_common():
        print(f"  {n:>3}  {reason}")

    if unstable:
        print(f"\n{len(unstable)} claim(s) gave inconsistent proposals across runs:")
        for c in unstable:
            print(f"  {c}")
        print("\nThese are the cases a single-run evaluation would score wrong.")


def _fmt(counter: Counter, total: int) -> str:
    return "   ".join(f"{k} {v}/{total}" for k, v in counter.most_common())


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG)
