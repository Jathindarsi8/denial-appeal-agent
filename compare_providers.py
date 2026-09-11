"""
Day 14: where do two providers actually disagree?

Yesterday, on CLM-100046, one provider appealed consistently and the other
closed the claim on 4 of 5 runs. Same claim, same verified authorization,
opposite conclusions. That is one claim out of five, and one comparison is an
anecdote.

This runs every case on every configured provider, several times each, and
reports three separate things that are easy to conflate:

  self-agreement   does a provider agree with itself across its own runs
  cross-agreement  do the providers agree with each other
  the split        when they disagree, which way does each one go

A provider that is stable and wrong looks identical to one that is stable and
right, so none of this says who is correct. It says where a human has to look.
That is what the resolutions table is for, and it is still empty.

Old runs in the log are NOT reused. The required checks now run before the
model's first turn and the authorization check is a real lookup, so anything
recorded before yesterday describes a different system.

Cost: cases x runs x roughly 2 calls. Five cases at three runs is about thirty
calls per provider. Fine on a large allowance, most of a day on a small one, so
run counts are per provider.

    python compare_providers.py                     3 runs each, both providers
    python compare_providers.py --groq 5 --gemini 2
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

from agent import DenialAppealAgent, DenialCodeLookup, ModelClient
from cases import ALL, LABELS

PAUSE = {"gemini": 45, "groq": 3}


def cases() -> list:
    return [(LABELS[c.claim_id], c) for c in ALL]


def run_provider(provider: str, model: str | None, runs: int) -> dict:
    results: dict[str, list[dict]] = defaultdict(list)
    pause = PAUSE.get(provider, 10)

    for label, case in cases():
        print(f"  {case.claim_id}  {label}")
        for i in range(runs):
            agent = DenialAppealAgent(
                code_lookup=DenialCodeLookup(),
                model=ModelClient(provider=provider, model=model),
                audit_log=False,   # comparison runs stay out of the real log
                resume=False,
                use_memory=False,  # history would differ between providers
            )
            try:
                state = agent.run(case)
            except Exception as exc:
                print(f"    run {i+1}  FAILED  {type(exc).__name__}")
                results[case.claim_id].append({"error": type(exc).__name__})
                continue

            j = state.judgment
            rec = {
                "final": state.decision.value if state.decision else None,
                "proposed": j.proposed_decision if j else None,
                "confidence": j.confidence if j else None,
                "stop_reason": state.stop_reason,
                "tools": list(state.tools_called),
            }
            results[case.claim_id].append(rec)
            conf = rec["confidence"]
            print(f"    run {i+1}  {str(rec['final']):<14} "
                  f"conf {conf if conf is None else f'{conf:.2f}'}")
            time.sleep(pause)
        print()

    return dict(results)


def unanimous(values: list) -> str | None:
    c = Counter(v for v in values if v is not None)
    return next(iter(c)) if len(c) == 1 else None


def report(all_results: dict[str, dict]) -> None:
    providers = list(all_results)
    print("=" * 74)
    print(f"{'claim':<14} " + "  ".join(f"{p:<22}" for p in providers))
    print("-" * 74)

    unstable: list[str] = []
    disagree: list[str] = []

    for label, case in cases():
        cid = case.claim_id
        row = [f"{cid:<14}"]
        finals_by_provider = {}

        for p in providers:
            runs = [r for r in all_results[p].get(cid, []) if "error" not in r]
            finals = [r["final"] for r in runs]
            counts = Counter(f for f in finals if f)
            finals_by_provider[p] = unanimous(finals)
            if not counts:
                row.append(f"{'(no runs)':<22}")
                continue
            summary = " ".join(f"{k[:12]} {v}" for k, v in counts.most_common())
            row.append(f"{summary:<22}")
            if len(counts) > 1:
                unstable.append(f"{cid} on {p}")

        print("  ".join(row))

        settled = [v for v in finals_by_provider.values() if v is not None]
        if len(settled) == len(providers) and len(set(settled)) > 1:
            disagree.append(cid)

    print()
    if unstable:
        print("A provider disagreed with itself on:")
        for u in unstable:
            print(f"  {u}")
        print("  Those cannot be compared across providers until they settle.")
    else:
        print("Every provider was self-consistent on every case.")

    print()
    if disagree:
        print("Providers were each internally consistent and disagreed with")
        print("each other on:")
        for d in disagree:
            print(f"  {d}")
        print()
        print("  These are the claims that need a human decision recorded.")
        print("  Both sides are stable, so neither is going to yield, and")
        print("  nothing in this project can currently say which is right.")
    else:
        print("No case where both providers were consistent and disagreed.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"compare-providers-{stamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved: {path}")


def main() -> None:
    plan: list[tuple[str, str | None, int]] = []

    if "--groq" in sys.argv:
        plan.append(("groq", None, int(sys.argv[sys.argv.index("--groq") + 1])))
    if "--gemini" in sys.argv:
        plan.append(("gemini", None,
                     int(sys.argv[sys.argv.index("--gemini") + 1])))
    if not plan:
        plan = [("groq", None, 3), ("gemini", None, 3)]

    print("Day 14: where do the providers disagree?\n")
    for provider, model, runs in plan:
        print(f"{provider}, {runs} run(s) per case\n")

    all_results = {}
    for provider, model, runs in plan:
        print(f"\n--- {provider} ---")
        all_results[provider] = run_provider(provider, model, runs)

    report(all_results)


if __name__ == "__main__":
    main()
