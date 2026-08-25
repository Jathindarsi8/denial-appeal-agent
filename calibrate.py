"""
Day 3: is the confidence floor calibrated to anything?

The guardrail layer escalates when the model reports confidence below 0.6.
That threshold was picked by feel. This measures whether it means the same
thing twice on one model, and whether it means the same thing across models.

Deliberately does NOT use tools. One model call per run, same prompt every
time, record what comes back. Tool-call paths vary run to run and would
confound the measurement. Also keeps it inside the free tier's 20 requests
per day per model.

    python calibrate.py                      # default model, 10 runs
    python calibrate.py gemini-3.7-flash 10  # explicit
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime

from agent import (
    SYSTEM_PROMPT,
    DenialCodeLookup,
    DenialRecord,
    ModelAction,
    ModelClient,
)

# The case from Day 2 that produced 0.98 on one model and 0.75 on another.
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

# Tools removed for this measurement — the model must judge from the record alone.
JUDGE_ONLY_PROMPT = SYSTEM_PROMPT.replace(
    "Each turn you return JSON, and you choose one of two actions.",
    "Return a single judgment. No tools are available on this run.",
)


def build_prompt(case: DenialRecord, code_meaning: str) -> str:
    return f"""\
Claim ID: {case.claim_id}
Payer: {case.payer}
Denied amount: ${case.amount:,.2f}
CARC: {case.carc}  RARC: {case.rarc or "N/A"}

Standard meaning of this denial code:
{code_meaning}

Payer's stated explanation:
{case.payer_explanation}

Documentation on file:
{case.documentation_summary}

No tools are available. Return the "judge" action only.
"""


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODEL", "gemini-3.6-flash")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    _, code_meaning = DenialCodeLookup().lookup(CASE.carc)
    client = ModelClient(model=model_name)

    messages = [
        {"role": "system", "content": JUDGE_ONLY_PROMPT},
        {"role": "user", "content": build_prompt(CASE, code_meaning)},
    ]

    print(f"model:  {model_name}")
    print(f"case:   {CASE.claim_id}  (CARC {CASE.carc}, contradicting evidence)")
    print(f"runs:   {runs}   temperature 0, identical prompt each time\n")

    results: list[dict] = []

    for i in range(1, runs + 1):
        try:
            action: ModelAction = client.step(messages)
        except Exception as exc:
            print(f"  run {i:>2}   FAILED  {type(exc).__name__}")
            results.append({"run": i, "error": type(exc).__name__})
            continue

        decision = action.proposed_decision or "none"
        confidence = action.confidence
        results.append({
            "run": i,
            "decision": decision,
            "confidence": confidence,
            "category": action.denial_category,
        })
        conf_str = f"{confidence:.2f}" if confidence is not None else "  --"
        print(f"  run {i:>2}   {decision:<16} confidence {conf_str}")

        # Free tier allows 5 requests/minute.
        if i < runs:
            time.sleep(13)

    print()
    summarize(model_name, runs, results)


def summarize(model_name: str, runs: int, results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    confidences = [r["confidence"] for r in ok if r["confidence"] is not None]
    decisions = [r["decision"] for r in ok]

    print("=" * 60)
    print(f"completed:  {len(ok)} of {runs}")

    if confidences:
        print(f"confidence: min {min(confidences):.2f}   "
              f"max {max(confidences):.2f}   "
              f"mean {statistics.mean(confidences):.2f}")
        if len(confidences) > 1:
            print(f"            spread {max(confidences) - min(confidences):.2f}   "
                  f"stdev {statistics.stdev(confidences):.3f}")

    if decisions:
        counts: dict[str, int] = {}
        for d in decisions:
            counts[d] = counts.get(d, 0) + 1
        spread = "  ".join(f"{k} {v}/{len(decisions)}" for k, v in sorted(counts.items()))
        print(f"decisions:  {spread}")

        if len(counts) > 1:
            print("\n  Same input, same temperature, more than one decision.")
        else:
            print("\n  Decision was stable across every run.")

    below_floor = [c for c in confidences if c < 0.6]
    if below_floor:
        print(f"\n  {len(below_floor)} of {len(confidences)} runs fell below the 0.6 floor.")
        print("  Those runs would have escalated on confidence alone.")
    elif confidences:
        print("\n  No run fell below the 0.6 floor.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"calibration-{model_name}-{stamp}.json"
    with open(path, "w") as f:
        json.dump({
            "model": model_name,
            "case": CASE.claim_id,
            "runs": runs,
            "results": results,
        }, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
