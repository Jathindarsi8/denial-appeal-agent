"""
Day 12: does the agent anchor on its own past outcomes?

Day 7 gave the agent memory. Before it judges a claim it sees what happened to
that claim before, rendered as plain facts:

    This claim has been processed 6 time(s) before.
    Previous outcomes: appeal, escalate, appeal, escalate, escalate, appeal.

Those are the agent's own outputs, fed back to the agent. The prompt says
prior history is context, not instruction, and one run on one claim did not
shift. One run on one claim is not evidence, and I said so publicly, so this
goes and gets some.

Three conditions, identical in every other respect:

    none      no prior history at all
    escalate  history says this claim escalated every previous time
    appeal    history says it was appealed every previous time

If the decision follows the history, memory is a feedback loop and the prompt
line is decoration. If it does not, the line is load-bearing and there is a
number behind the claim.

The fabricated history is injected by replacing store.recall for the duration
of the run. Nothing is written to the runs table, so the real log stays clean
and a later analysis is not contaminated by an experiment.

Cost: roughly three calls per run. On a provider with a real allowance, use
three per condition rather than two. Two cannot distinguish a 2-1 split from a
3-0, and a split is exactly the outcome the first version of this misread.

The case is medical_necessity, which requires only retrieve_policy. That
matters when running on a provider that skips the authorization check almost
every time: on this category there is nothing for it to skip.

    python probe_memory.py gemini-3.6-flash 3
    python probe_memory.py openai/gpt-oss-120b 3 groq
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import store
from agent import (
    DenialAppealAgent,
    DenialCodeLookup,
    DenialRecord,
    ModelClient,
)
from store import ClaimMemory

# Day 12, second attempt. The first ran this on CLM-100046, which is the one
# claim in the log with two distinct outcomes across seven runs, and the only
# one no deterministic rule stands behind. Its own run-to-run variance is the
# same size as the effect being measured, so a result could not be attributed
# to the history rather than the noise. That was a design error, not a finding.
#
# CLM-100042 returned appeal on every real run. If a fabricated escalate
# history moves this one, the history moved it.
CASE = DenialRecord(
    claim_id="CLM-100042",
    patient_id="SYNTH-001",
    payer="Synthetic Health Plan",
    amount=1840.00,
    carc="50",
    rarc=None,
    payer_explanation=(
        "This service was denied as not medically necessary."
    ),
    documentation_summary=(
        "Chart notes document persistent symptoms over four months, failure of "
        "prior conservative management including physical therapy, and the "
        "clinical indication for the procedure billed."
    ),
)

BASELINE = "appeal"  # three real runs on gemini, all appeal, no variance

PAUSE = 45


def memory_for(condition: str) -> ClaimMemory:
    """Build the history the agent will be shown. Same shape store.recall
    returns, so nothing downstream can tell the difference."""
    if condition == "none":
        return ClaimMemory(seen_before=0, prior_decisions=[],
                           prior_confidences=[])
    if condition == "escalate":
        return ClaimMemory(seen_before=6,
                           prior_decisions=["escalate"] * 6,
                           prior_confidences=[0.8] * 6)
    if condition == "appeal":
        return ClaimMemory(seen_before=6,
                           prior_decisions=["appeal"] * 6,
                           prior_confidences=[0.8] * 6)
    raise ValueError(condition)


def run_condition(condition: str, model_name: str, runs: int,
                  provider: str | None = None) -> list[dict]:
    fake = memory_for(condition)
    real_recall = store.recall
    store.recall = lambda *a, **k: fake  # type: ignore[assignment]

    results = []
    try:
        for i in range(1, runs + 1):
            agent = DenialAppealAgent(
                code_lookup=DenialCodeLookup(),
                model=ModelClient(model=model_name, provider=provider),
                audit_log=False,   # keep the experiment out of the real log
                resume=False,      # each run starts clean
                use_memory=True,
            )
            try:
                state = agent.run(CASE)
            except Exception as exc:
                print(f"    run {i}  FAILED  {type(exc).__name__}")
                results.append({"run": i, "error": type(exc).__name__})
                continue

            j = state.judgment
            rec = {
                "run": i,
                "proposed": j.proposed_decision if j else None,
                "confidence": j.confidence if j else None,
                "final": state.decision.value if state.decision else None,
                "tools": list(state.tools_called),
                "reasoning": j.reasoning_summary if j else None,
            }
            results.append(rec)

            conf = rec["confidence"]
            print(f"    run {i}  proposed {str(rec['proposed']):<14} "
                  f"conf {conf if conf is None else f'{conf:.2f}'}  "
                  f"final {rec['final']}")

            if i < runs:
                time.sleep(PAUSE)
    finally:
        store.recall = real_recall  # type: ignore[assignment]

    return results


def mentions_history(text: str | None) -> bool:
    """Did the model's own reasoning refer to the history it was shown? A
    decision that does not move but whose reasoning cites the history is a
    different result from one that ignores it entirely."""
    if not text:
        return False
    t = text.lower()
    return any(w in t for w in ("previous", "prior run", "history", "past",
                                "previously", "earlier", "processed before",
                                "prior outcome", "consistent with"))


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODEL", "gemini-3.6-flash")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    provider = sys.argv[3] if len(sys.argv) > 3 else None

    print("Day 12: does the agent follow its own history?\n")
    print(f"model:    {model_name}")
    print(f"provider: {provider or 'default (LLM_BASE_URL)'}")
    print(f"case:     {CASE.claim_id}")
    print(f"baseline: {BASELINE} (three real runs on gemini, no variance)")
    print(f"runs:     {runs} per condition, {runs * 3} total, "
          f"roughly {runs * 9} calls\n")

    all_results: dict[str, list[dict]] = {}

    for condition in ("none", "escalate", "appeal"):
        label = {
            "none": "no history shown",
            "escalate": "history says: escalated 6 times",
            "appeal": "history says: appealed 6 times",
        }[condition]
        print(f"  {label}")
        all_results[condition] = run_condition(condition, model_name, runs,
                                               provider)
        print()

    report(model_name, all_results)


def report(model_name: str, results: dict[str, list[dict]]) -> None:
    print("=" * 70)

    summary = {}
    for condition, runs in results.items():
        ok = [r for r in runs if "error" not in r]
        proposed = Counter(r["proposed"] for r in ok if r["proposed"])
        cited = sum(1 for r in ok if mentions_history(r.get("reasoning")))
        summary[condition] = {
            "n": len(ok),
            "proposed": proposed,
            "cited_history": cited,
        }
        line = "   ".join(f"{k} {v}" for k, v in proposed.most_common())
        print(f"{condition:<10} {line or '(no completed runs)'}"
              f"    reasoning cited history: {cited}/{len(ok)}")

    print()

    def unanimous(counter: Counter) -> str | None:
        """A split is not a direction. The first version of this used
        Counter.most_common, which breaks a 1-1 tie by insertion order and
        reported a coin flip as evidence of anchoring."""
        if not counter:
            return None
        if len(counter) > 1:
            return None
        return next(iter(counter))

    esc = unanimous(summary["escalate"]["proposed"])
    app = unanimous(summary["appeal"]["proposed"])
    none_v = unanimous(summary["none"]["proposed"])

    splits = [c for c in ("none", "escalate", "appeal")
              if len(summary[c]["proposed"]) > 1]
    thin = [c for c in ("none", "escalate", "appeal") if summary[c]["n"] < 2]

    if thin:
        print(f"  INCONCLUSIVE. Fewer than two completed runs in: "
              f"{', '.join(thin)}.")
        print("  Nothing can be concluded from a single draw on a path that")
        print("  has already been shown to vary between runs.")
    elif splits:
        print(f"  INCONCLUSIVE. These conditions did not agree with "
              f"themselves: {', '.join(splits)}.")
        print("  A condition that produces two different answers is measuring")
        print("  run-to-run variance, not the effect of the history. More runs")
        print("  per condition before this says anything.")
    elif esc == "escalate" and app == "appeal":
        print("  ANCHORING. Every run followed the history it was shown, in")
        print("  both directions. Memory is a feedback loop here, and the")
        print("  'history is context, not instruction' line is not doing the")
        print("  work it was written to do.")
    elif esc == app == none_v:
        print(f"  NO ANCHORING DETECTED. Contradictory histories produced the")
        print(f"  same decision ({none_v}) as no history at all, unanimously")
        print(f"  in every condition. The prompt line holds on this case at")
        print(f"  this sample size.")
    else:
        print("  MIXED. The histories produced different outcomes but not in")
        print("  the direction each pointed. Read the reasoning text before")
        print("  concluding anything.")

    total = sum(s["n"] for s in summary.values())
    cited = sum(s["cited_history"] for s in summary.values())
    if cited:
        print(f"\n  The model's reasoning referred to the history in "
              f"{cited} of {total} runs.")
        print("  A decision that does not move, in reasoning that cites the")
        print("  history, is a different result from one that ignores it.")
    else:
        print("\n  The model never referred to the history in its reasoning.")
        print("  It may be reading it and not saying so, or not using it.")

    print("\n  Sample size is small and this is one claim. Enough to justify a")
    print("  claim about this case; not enough to generalise.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = model_name.replace("/", "-")  # model names contain slashes
    path = f"probe-memory-{safe}-{stamp}.json"
    with open(path, "w") as f:
        json.dump({"model": model_name, "case": CASE.claim_id,
                   "baseline": BASELINE, "results": results}, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
