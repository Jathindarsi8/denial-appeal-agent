"""
Day 6: is the tool-skipping my fault?

Day 5 found that on CLM-100046, three of four completed runs called no tools at
all, judged from the claim record alone, and reported 0.95 confidence. Same
model, same input, temperature 0.

Before calling that model instability, check the prompt. It currently says:

    - Call a tool only when the answer would actually change your assessment.

That is an instruction to skip retrieval when the model judges it unnecessary.
The skipping runs may have been obeying it.

This script removes that one line and nothing else, then runs the same case the
same number of times. If the skip rate drops, the instruction caused it and the
fix is a prompt change, not a model problem. If it doesn't, the instability is
real and deeper.

Baseline is NOT re-run here. Day 5 already recorded four runs of this case on
gemini-3.7-flash with the original prompt, and spending quota to reproduce a
result already on disk is how you run out of calls. Compare against
runs/runs.jsonl.

    python probe_prompt.py gemini-3.7-flash 4
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime

import agent as agent_module
from agent import (
    DenialAppealAgent,
    DenialCodeLookup,
    DenialRecord,
    ModelClient,
    SYSTEM_PROMPT,
)

# The single line under test.
TARGET_LINE = "- Call a tool only when the answer would actually change your assessment.\n"

CASE = DenialRecord(
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

PAUSE_BETWEEN_RUNS = 45


def build_variant() -> str:
    """Original prompt with exactly one line removed."""
    if TARGET_LINE not in SYSTEM_PROMPT:
        raise SystemExit(
            "Could not find the target line in SYSTEM_PROMPT. It may have been "
            "edited. Check agent.py and update TARGET_LINE to match exactly."
        )
    return SYSTEM_PROMPT.replace(TARGET_LINE, "")


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODEL", "gemini-3.7-flash")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    variant = build_variant()

    print("Day 6 probe: does one prompt line cause the tool-skipping?\n")
    print(f"model:   {model_name}")
    print(f"case:    {CASE.claim_id}  (CARC {CASE.carc})")
    print(f"runs:    {runs}")
    print(f"removed: {TARGET_LINE.strip()}")
    print(f"prompt:  {len(SYSTEM_PROMPT)} chars -> {len(variant)} chars\n")

    # Swap the prompt the agent module uses. Everything else is untouched.
    agent_module.SYSTEM_PROMPT = variant

    results: list[dict] = []

    for i in range(1, runs + 1):
        a = DenialAppealAgent(
            code_lookup=DenialCodeLookup(),
            model=ModelClient(model=model_name),
            audit_log=False,  # keep the probe out of the main run log
        )

        try:
            state = a.run(CASE)
        except Exception as exc:
            print(f"  run {i:>2}   FAILED  {type(exc).__name__}")
            results.append({"run": i, "error": type(exc).__name__})
            continue

        j = state.judgment
        rec = {
            "run": i,
            "tools_called": list(state.tools_called),
            "retrieved": bool(state.observations),
            "proposed_decision": j.proposed_decision if j else None,
            "confidence": j.confidence if j else None,
            "final_decision": state.decision.value if state.decision else None,
            "stop_reason": state.stop_reason,
        }
        results.append(rec)

        conf = rec["confidence"]
        conf_str = f"{conf:.2f}" if conf is not None else "  --"
        tools = ",".join(rec["tools_called"]) or "NONE"
        print(
            f"  run {i:>2}   tools [{tools:<45}] "
            f"conf {conf_str}   {rec['final_decision']}"
        )

        if i < runs:
            time.sleep(PAUSE_BETWEEN_RUNS)

    print()
    report(model_name, runs, results)


def report(model_name: str, runs: int, results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("no runs completed")
        return

    skipped = [r for r in ok if not r["retrieved"]]
    confidences = [r["confidence"] for r in ok if r["confidence"] is not None]

    print("=" * 70)
    print(f"completed:      {len(ok)} of {runs}")
    print(f"skipped tools:  {len(skipped)} of {len(ok)}")

    if confidences:
        line = f"confidence:     {min(confidences):.2f} to {max(confidences):.2f}"
        if len(confidences) > 1:
            line += f"   stdev {statistics.stdev(confidences):.3f}"
        print(line)

    print("\nDay 5 baseline, same case, same model, original prompt:")
    print("  skipped tools:  3 of 4")

    print()
    if not skipped:
        print("  No run skipped retrieval. The instruction was causing it.")
        print("  This is a prompt bug, not model instability. Fix the prompt.")
    elif len(skipped) < len(ok) / 2:
        print("  Skipping dropped but did not stop. The instruction contributes,")
        print("  but something else is going on too.")
    else:
        print("  Still skipping at roughly the same rate. The instruction is not")
        print("  the cause, and the instability is real. Needs a different fix:")
        print("  make retrieval a step in the loop rather than a model choice.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"probe-prompt-{model_name}-{stamp}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "case": CASE.claim_id,
                "line_removed": TARGET_LINE.strip(),
                "runs": runs,
                "baseline_day5_skipped": "3 of 4",
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
