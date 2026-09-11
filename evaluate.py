"""
Day 15: scoring the agent against ground truth.

Day 4 established that one run tells you nothing: the same claim can come back
appeal or escalate. Day 14 quantified it, at 17 appeals in 40 runs on the one
claim no rule stands behind. So a golden set scored once per case measures a
single draw from a distribution and reports it as accuracy.

This scores two things, and the gap between them is the point.

  pass@1   of all runs, what fraction were correct. The number a demo shows.
  pass^k   of the cases, what fraction were correct on EVERY run. The number
           that says whether you could ship it.

A case that is right 3 times in 5 contributes 0.6 to pass@1 and 0 to pass^k.
That is the correct treatment. A claims agent that files the right appeal most
of the time is not a claims agent that files the right appeal.

Memory is off for every run. The golden labels live in the resolutions table,
the agent can read that table, and a guardrail escalates when it contradicts a
recorded human decision. Leaving memory on would mean testing the agent against
answers it has been handed.

    python evaluate.py groq 5
    python evaluate.py gemini 2
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime

import store
from agent import DenialAppealAgent, DenialCodeLookup, ModelClient
from golden import GOLDEN
from cases import BY_ID

PAUSE = {"gemini": 45, "groq": 3}


def cases_by_id() -> dict:
    return BY_ID


def evaluate(provider: str, runs: int) -> dict:
    available = cases_by_id()
    pause = PAUSE.get(provider, 10)
    results: dict[str, dict] = {}

    for label in GOLDEN:
        case = available.get(label.claim_id)
        if case is None:
            print(f"  {label.claim_id}  no matching case in run_cases, skipped")
            continue

        print(f"  {label.claim_id}   expected {label.decision}")
        decisions: list[str] = []

        for i in range(runs):
            agent = DenialAppealAgent(
                code_lookup=DenialCodeLookup(),
                model=ModelClient(provider=provider),
                audit_log=False,
                resume=False,
                use_memory=False,  # the answers are in the resolutions table
            )
            try:
                state = agent.run(case)
                got = state.decision.value if state.decision else None
            except Exception as exc:
                print(f"    run {i+1}  FAILED  {type(exc).__name__}")
                decisions.append(None)
                continue

            decisions.append(got)
            mark = "ok " if got == label.decision else "XX "
            print(f"    run {i+1}  {mark} {got}")
            time.sleep(pause)

        ok = [d for d in decisions if d is not None]
        correct = sum(1 for d in ok if d == label.decision)
        results[label.claim_id] = {
            "expected": label.decision,
            "decisions": decisions,
            "runs": len(ok),
            "correct": correct,
            "pass_at_1": correct / len(ok) if ok else 0.0,
            "pass_all": bool(ok) and correct == len(ok),
        }
        print()

    return results


def report(provider: str, runs: int, results: dict) -> None:
    print("=" * 72)
    print(f"{'claim':<14}{'expected':<16}{'correct':<12}{'pass@1':<10}"
          f"{'every run'}")
    print("-" * 72)

    for cid, r in results.items():
        every = "yes" if r["pass_all"] else "NO"
        ratio = f"{r['correct']}/{r['runs']}"
        print(f"{cid:<14}{r['expected']:<16}{ratio:<12}"
              f"{r['pass_at_1']:.0%}      {every}")

    total_runs = sum(r["runs"] for r in results.values())
    total_correct = sum(r["correct"] for r in results.values())
    all_right = sum(1 for r in results.values() if r["pass_all"])

    pass_at_1 = total_correct / total_runs if total_runs else 0.0
    pass_k = all_right / len(results) if results else 0.0

    print()
    print(f"pass@1   {pass_at_1:.0%}   "
          f"({total_correct} correct of {total_runs} runs)")
    print(f"pass^{runs}   {pass_k:.0%}   "
          f"({all_right} of {len(results)} cases correct on every run)")

    print()
    gap = pass_at_1 - pass_k
    if gap > 0.15:
        print("  The gap is the finding. Most runs are right and some cases")
        print("  are not reliably right, which are different claims about the")
        print("  same system. Only the second one decides whether it ships.")
    elif pass_k == 1.0:
        print("  Every case correct on every run at this sample size. That is")
        print("  a statement about these cases and this many runs, not about")
        print("  the system.")
    else:
        print("  pass@1 and pass^k are close, so the failures are consistent")
        print("  rather than random. A consistent failure is easier to fix.")

    unstable = [cid for cid, r in results.items()
                if 0 < r["correct"] < r["runs"]]
    if unstable:
        print("\n  Cases that were right on some runs and wrong on others:")
        for cid in unstable:
            r = results[cid]
            c = Counter(d for d in r["decisions"] if d)
            spread = "  ".join(f"{k} {v}" for k, v in c.most_common())
            print(f"    {cid}  {spread}")
        print("  These are the ones a single-run evaluation would report as")
        print("  either a pass or a failure, depending on the draw.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"evaluate-{provider}-{stamp}.json"
    with open(path, "w") as f:
        json.dump({"provider": provider, "runs_per_case": runs,
                   "pass_at_1": pass_at_1, "pass_k": pass_k,
                   "results": results}, f, indent=2)
    print(f"\nsaved: {path}")


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "groq"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    store.init()
    with store.connect() as conn:
        n = conn.execute("SELECT COUNT(*) n FROM resolutions").fetchone()["n"]
    if n == 0:
        print("No resolutions recorded. Run: python golden.py record")
        return

    print(f"Evaluating against {len(GOLDEN)} labelled claims\n")
    print(f"provider: {provider}")
    print(f"runs:     {runs} per case, {runs * len(GOLDEN)} total\n")

    results = evaluate(provider, runs)
    report(provider, runs, results)


if __name__ == "__main__":
    main()
