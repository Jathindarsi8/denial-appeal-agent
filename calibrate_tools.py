"""
Multi-run stability harness. Drives the real agent, tools enabled.

Day 4 ran this on CLM-100045 and found the model proposed escalate three times
and appeal twice, same model, temperature 0. That case is a non-covered charge,
so the category rule caught the appeals regardless — the instability never
reached a real decision.

CLM-100046 is the sharper test. authorization_missing is NOT in
NEVER_AUTO_APPEAL, so nothing deterministic stands behind the model. If this
one flips across runs, an appeal is genuinely authorized on some runs and not
on others, and the guardrails don't save you.

Cost: one run is several API calls, not one — a model turn per tool request
plus the final judgment. On a 20/day cap that means five runs, not ten.

    python calibrate_tools.py                            # default model, 5 runs, 100046
    python calibrate_tools.py gemini-3.7-flash 5         # explicit model and count
    python calibrate_tools.py gemini-3.7-flash 5 100045  # the day 4 case
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime

from agent import (
    CONFIDENCE_FLOOR,
    DenialAppealAgent,
    DenialCodeLookup,
    DenialRecord,
    ModelClient,
)

# Non-covered charge with contradicting prior auth. The category rule always
# catches this one, so model instability never changes the outcome.
CASE_100045 = DenialRecord(
    claim_id="CLM-100045",
    patient_id="SYNTH-004",
    payer="Synthetic Health Plan",
    amount=2450.00,
    carc="96",
    rarc="N130",
    payer_explanation="This service is not covered under the member's benefit plan.",
    documentation_summary=(
        "Prior authorization reference PA-88213 was approved by the payer on "
        "2026-06-02 for this exact procedure code. The member's benefit summary "
        "lists the service as covered when medically necessary. The treating "
        "clinician documented the indication, and the payer's own approval "
        "letter is on file."
    ),
)

# Authorization missing, PA referenced in the notes. Nothing deterministic
# blocks an appeal here — the model's judgment is load-bearing.
CASE_100046 = DenialRecord(
    claim_id="CLM-100046",
    patient_id="SYNTH-005",
    payer="Synthetic Health Plan",
    amount=1375.00,
    carc="197",
    rarc=None,
    payer_explanation=(
        "Precertification was not obtained prior to the service being rendered."
    ),
    documentation_summary=(
        "Scheduling notes reference prior authorization PA-77104 obtained before "
        "the date of service. The authorization number was not included on the "
        "original claim submission."
    ),
)

CASES = {"100045": CASE_100045, "100046": CASE_100046}

# Five requests per minute on the free tier. A run makes several calls, so
# pause between runs rather than between calls.
PAUSE_BETWEEN_RUNS = 45


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODEL", "gemini-3.6-flash")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    case_key = sys.argv[3] if len(sys.argv) > 3 else "100046"

    if case_key not in CASES:
        print(f"unknown case '{case_key}'. options: {', '.join(CASES)}")
        return

    case = CASES[case_key]

    print(f"model:  {model_name}")
    print(f"case:   {case.claim_id}  (CARC {case.carc})")
    print(f"runs:   {runs}   tools ENABLED, temperature 0")
    print(f"floor:  {CONFIDENCE_FLOOR}\n")

    results: list[dict] = []

    for i in range(1, runs + 1):
        agent = DenialAppealAgent(
            code_lookup=DenialCodeLookup(),
            model=ModelClient(model=model_name),
        )

        try:
            state = agent.run(case)
        except Exception as exc:
            print(f"  run {i:>2}   FAILED  {type(exc).__name__}")
            results.append({"run": i, "error": type(exc).__name__})
            continue

        j = state.judgment
        record = {
            "run": i,
            "final_decision": state.decision.value if state.decision else None,
            "stop_reason": state.stop_reason,
            "tools_called": list(state.tools_called),
            "steps_taken": state.steps_taken,
            "proposed_decision": j.proposed_decision if j else None,
            "confidence": j.confidence if j else None,
            "denial_category": j.denial_category if j else None,
            "appeal_drafted": state.appeal_draft is not None,
            "trace": list(state.trace),
        }
        results.append(record)

        conf = record["confidence"]
        conf_str = f"{conf:.2f}" if conf is not None else "  --"
        tools = ",".join(record["tools_called"]) or "none"
        print(
            f"  run {i:>2}   proposed {str(record['proposed_decision']):<14} "
            f"conf {conf_str}   final {str(record['final_decision']):<14} "
            f"tools [{tools}]"
        )

        if i < runs:
            time.sleep(PAUSE_BETWEEN_RUNS)

    print()
    summarize(model_name, case, runs, results)


def summarize(model_name: str, case: DenialRecord, runs: int, results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    confidences = [r["confidence"] for r in ok if r["confidence"] is not None]

    print("=" * 68)
    print(f"completed:  {len(ok)} of {runs}")

    if confidences:
        print(
            f"confidence: min {min(confidences):.2f}   "
            f"max {max(confidences):.2f}   "
            f"mean {statistics.mean(confidences):.2f}"
        )
        if len(confidences) > 1:
            print(
                f"            spread {max(confidences) - min(confidences):.2f}   "
                f"stdev {statistics.stdev(confidences):.3f}"
            )

    _tally(ok, "proposed_decision", "proposed:  ")
    _tally(ok, "final_decision", "final:     ")

    tool_sets = ["+".join(r["tools_called"]) or "none" for r in ok]
    if tool_sets:
        counts: dict[str, int] = {}
        for t in tool_sets:
            counts[t] = counts.get(t, 0) + 1
        line = "   ".join(f"{k} {v}/{len(tool_sets)}" for k, v in sorted(counts.items()))
        print(f"tools:      {line}")

    drafted = sum(1 for r in ok if r.get("appeal_drafted"))
    finals = set(r["final_decision"] for r in ok)

    print()
    if len(finals) > 1:
        print(f"  THE FINAL DECISION MOVED: {', '.join(sorted(str(f) for f in finals))}")
        print(f"  An appeal was drafted on {drafted}/{len(ok)} runs of the same claim.")
        print("  Nothing deterministic caught this. The model's judgment decided it.")
    elif drafted == len(ok) and ok:
        print(f"  Stable: an appeal was authorized on all {len(ok)} runs.")
    elif ok:
        print(f"  Stable: every run landed on {list(finals)[0]}.")

    below = [c for c in confidences if c < CONFIDENCE_FLOOR]
    if below:
        print(f"\n  {len(below)} of {len(confidences)} runs fell below the {CONFIDENCE_FLOOR} floor.")
    elif confidences:
        print(f"\n  No run fell below the {CONFIDENCE_FLOOR} floor.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"calibration-tools-{case.claim_id}-{model_name}-{stamp}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "case": case.claim_id,
                "carc": case.carc,
                "tools_enabled": True,
                "confidence_floor": CONFIDENCE_FLOOR,
                "runs": runs,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nsaved: {path}")


def _tally(ok: list[dict], key: str, label: str) -> None:
    values = [r[key] for r in ok if r[key] is not None]
    if not values:
        return
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    line = "   ".join(f"{k} {v}/{len(values)}" for k, v in sorted(counts.items()))
    print(f"{label}{line}")


if __name__ == "__main__":
    main()
