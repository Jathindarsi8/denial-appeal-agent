"""
Day 4: the missing cell.

Day 3 measured confidence with tools stripped out — one model call per run,
nothing to read. Ten runs, 0.95 every time, stdev 0.000.

This measures the other path: the real agent, tools available, deciding for
itself what to look up. Same case, same model, same temperature. The question
is whether the 0.75 from Sunday was stable or a single draw.

Unlike calibrate.py, this does NOT bypass the agent. It calls agent.run(),
so what gets measured is the actual production path including guardrails.

Cost note: one run here is several API calls, not one — a model turn per tool
request plus the final judgment. On the free tier's 20/day cap that means five
runs, not ten. Don't raise it without checking the arithmetic.

    python calibrate_tools.py                     # default model, 5 runs
    python calibrate_tools.py gemini-3.6-flash 5  # explicit
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

# Same case as calibrate.py, so the two results are directly comparable.
CASE = DenialRecord(
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

# Five requests per minute on the free tier. A run makes several calls, so
# pause between runs rather than between calls.
PAUSE_BETWEEN_RUNS = 45


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODEL", "gemini-3.6-flash")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"model:  {model_name}")
    print(f"case:   {CASE.claim_id}  (CARC {CASE.carc}, contradicting evidence)")
    print(f"runs:   {runs}   tools ENABLED, temperature 0")
    print(f"floor:  {CONFIDENCE_FLOOR}\n")

    results: list[dict] = []

    for i in range(1, runs + 1):
        agent = DenialAppealAgent(
            code_lookup=DenialCodeLookup(),
            model=ModelClient(model=model_name),
        )

        try:
            state = agent.run(CASE)
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
    summarize(model_name, runs, results)


def summarize(model_name: str, runs: int, results: list[dict]) -> None:
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
        if len(counts) > 1:
            print("\n  The agent did not always look up the same things.")
        else:
            print("\n  The agent retrieved the same evidence on every run.")

    no_evidence = [r for r in ok if not r["tools_called"]]
    if no_evidence:
        print(f"\n  {len(no_evidence)} run(s) reached a judgment without retrieving anything.")
        for r in no_evidence:
            print(f"    run {r['run']}: {r['proposed_decision']} at {r['confidence']}")

    below = [c for c in confidences if c < CONFIDENCE_FLOOR]
    if below:
        print(f"\n  {len(below)} of {len(confidences)} runs fell below the {CONFIDENCE_FLOOR} floor.")
    elif confidences:
        print(f"\n  No run fell below the {CONFIDENCE_FLOOR} floor.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"calibration-tools-{model_name}-{stamp}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "case": CASE.claim_id,
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
