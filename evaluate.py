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


# Day 16. Scoring the final decision cannot tell "the model was right" from
# "the model was wrong and a rule caught it". Those are different systems with
# the same score. CIVI's point applies here: record which control produced the
# outcome, not only whether the outcome was correct.
#
# Two of ARISE's four modes are structurally impossible in this system on a
# known denial code. The required checks run before the model's first turn, so
# it cannot bypass search and cannot fail to retrieve. What is left is what
# the model does with evidence it already has, and what the rules do about it.
MODE_LABEL = {
    "clean": "model right, rules agreed",
    "rescued": "model WRONG, rules caught it",
    "leaked": "model wrong, RULES LET IT THROUGH",
    "overblocked": "model right, RULES OVERRULED IT",
    "both_wrong": "model wrong, rules wrong differently",
    "no_judgment": "stopped before the model judged",
}


# Day 17. Not every wrong answer costs the same, and scoring them as if they
# do misreports what the system is worth.
#
# The day 17 rule fixed CLM-100045 outright and changed the shape of the
# failures on CLM-100046: twelve runs that used to close a claim with a
# verified authorization now escalate it instead. Scored flat, that looks like
# a small regression. Operationally it is not remotely the same event. One
# ends with $1,375 quietly never collected and nobody aware of it. The other
# ends with a claim on a reviewer's desk and the appeal filed.
#
# A guardrail can only refuse a proposal, never promote one. Turning a bad
# close into an escalation is the most a rule can do, and the metric should
# say so rather than treating it as the same failure.
SEVERITY = {
    # (expected, actual): how bad, and who finds out
    ("appeal", "escalate"): ("recoverable", "a human sees it and can still file"),
    ("appeal", "do_not_appeal"): ("silent", "money never collected, nobody knows"),
    ("escalate", "appeal"): ("recoverable", "the appeal is rejected and someone sees it"),
    ("escalate", "do_not_appeal"): ("silent", "closed without the review the rule requires"),
    ("do_not_appeal", "appeal"): ("recoverable", "wasted effort, visible when rejected"),
    ("do_not_appeal", "escalate"): ("recoverable", "wasted reviewer time, visible"),
}

SEVERITY_ORDER = ["correct", "recoverable", "silent"]


def severity(expected: str, actual: str | None) -> str:
    if actual == expected:
        return "correct"
    if actual is None:
        return "silent"
    return SEVERITY.get((expected, actual), ("recoverable", ""))[0]


def classify(expected: str, proposed: str | None, final: str | None) -> str:
    if proposed is None:
        # Escalated before the model reached a judgment: an unmapped code, or
        # a required check with no implementation. Correct or not, the model
        # was never involved.
        return "no_judgment"

    model_right = proposed == expected
    final_right = final == expected

    if model_right and final_right:
        return "clean"
    if not model_right and final_right:
        return "rescued"
    if model_right and not final_right:
        return "overblocked"
    if proposed == final:
        return "leaked"
    return "both_wrong"


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
        runs_detail: list[dict] = []

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
                proposed = (state.judgment.proposed_decision
                            if state.judgment else None)
                stop = state.stop_reason
            except Exception as exc:
                print(f"    run {i+1}  FAILED  {type(exc).__name__}")
                runs_detail.append({"error": type(exc).__name__})
                continue

            mode = classify(label.decision, proposed, got)
            sev = severity(label.decision, got)
            runs_detail.append({
                "final": got,
                "proposed": proposed,
                "stop_reason": stop,
                "mode": mode,
                "severity": sev,
            })
            mark = "ok " if got == label.decision else "XX "
            print(f"    run {i+1}  {mark} {str(got):<14} {MODE_LABEL[mode]}")
            time.sleep(pause)

        ok = [r for r in runs_detail if "error" not in r]
        correct = sum(1 for r in ok if r["final"] == label.decision)
        results[label.claim_id] = {
            "expected": label.decision,
            "decisions": [r["final"] for r in ok],
            "detail": runs_detail,
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

    # ---- what actually produced each outcome
    modes = Counter()
    for r in results.values():
        for d in r.get("detail", []):
            if "mode" in d:
                modes[d["mode"]] += 1

    print("what produced each outcome:")
    for mode in ("clean", "rescued", "leaked", "overblocked", "both_wrong",
                 "no_judgment"):
        if modes[mode]:
            print(f"  {modes[mode]:>3}  {MODE_LABEL[mode]}")

    total = sum(modes.values())
    model_alone = modes["clean"] + modes["overblocked"]
    if total:
        print(f"\n  The model proposed the correct answer on "
              f"{model_alone}/{total} runs ({model_alone/total:.0%}).")
        print(f"  The system produced the correct answer on "
              f"{modes['clean'] + modes['rescued']}/{total} runs "
              f"({(modes['clean'] + modes['rescued'])/total:.0%}).")
        gap = modes["rescued"]
        if gap:
            print(f"\n  {gap} run(s) were correct only because a rule refused")
            print(f"  what the model proposed. Score the model alone and those")
            print(f"  are failures.")
        if modes["leaked"]:
            print(f"\n  {modes['leaked']} run(s) were wrong and nothing caught")
            print(f"  them. These are the ones worth a new rule.")
        if modes["overblocked"]:
            print(f"\n  {modes['overblocked']} run(s) had a correct proposal "
                  f"refused by a rule.")
            print(f"  Safe, and it costs a decision that did not need a human.")

    # ---- what a failure actually costs
    sevs = Counter()
    for r in results.values():
        for d in r.get("detail", []):
            if "severity" in d:
                sevs[d["severity"]] += 1

    total_sev = sum(sevs.values())
    if total_sev:
        print("\nwhat the failures cost:")
        print(f"  {sevs['correct']:>3}  correct")
        print(f"  {sevs['recoverable']:>3}  wrong, but a human finds out")
        print(f"  {sevs['silent']:>3}  wrong, and nobody finds out")

        print(f"\n  silent failure rate: "
              f"{sevs['silent']}/{total_sev} ({sevs['silent']/total_sev:.0%})")
        if sevs["silent"] == 0:
            print("  Every remaining failure lands in front of a person.")
            print("  That is the most a refuse-only guardrail layer can do:")
            print("  it cannot promote a bad proposal into the right answer,")
            print("  only stop it from becoming an action.")
        else:
            print("  These are the ones that cost money nobody ever sees.")

        for r in results.values():
            for d in r.get("detail", []):
                if d.get("severity") == "silent":
                    pair = (r["expected"], d["final"])
                    note = SEVERITY.get(pair, ("", ""))[1]
                    print(f"    expected {pair[0]}, got {pair[1]}: {note}")
                    break

    print()
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
