"""
Day 15: ground truth, and being honest about where it comes from.

Every finding so far ends the same way. The agent appeals this claim 42% of the
time without memory and 75% with it, and nothing in the project can say which
is closer to right. The `resolutions` table has been empty since day 7.

This fills it. Carefully, because a golden set that asserts more than it can
defend is worse than no golden set at all: every later evaluation inherits its
mistakes and reports them as accuracy.

The rule applied to every label here: *a decision goes in the set only if a
published rule determines it.* Where the answer turns on clinical judgment, the
correct answer is `escalate`, which is also what the system should do. That is
not a dodge. In claims work "a human decides this" is a real and common correct
answer, and a set that pretends otherwise measures the wrong thing.

What that excludes: whether a specific patient's documentation genuinely
establishes medical necessity. I am not qualified to label that and neither is
the model. What the set can assert is whether the record contains the elements
the payer's own policy requires, which is a documentation question, not a
clinical one.

Two of these labels were only defensible after day 13. Before the authorization
check became a real lookup, CLM-100046 rested on the claim notes asserting an
authorization existed. Now the system of record confirms it exists, is approved,
covers the billed procedure, and falls inside the window, and AU-07 says an
appeal is accepted on exactly that. The label follows from the evidence rather
than from an opinion about it.

    python golden.py            show the set and its justifications
    python golden.py record     write it to the resolutions table
    python golden.py clear      remove it again
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import store

LABELLED_BY = "golden-set-v1"


@dataclass
class Label:
    claim_id: str
    decision: str       # appeal / do_not_appeal / escalate
    rule: str           # the published rule that determines it
    reasoning: str
    certainty: str      # "rule" or "judgment"


GOLDEN: list[Label] = [
    Label(
        claim_id="CLM-100042",
        decision="appeal",
        rule="Policy MN-04, what a first-level appeal must contain",
        reasoning=(
            "MN-04 requires a first-level medical necessity appeal to establish "
            "the working diagnosis and severity, prior conservative management "
            "and its outcome, and the reasoning connecting the two to the "
            "service billed. The record contains all three. Whether a reviewer "
            "would ultimately agree is a clinical question; whether the record "
            "meets the policy's documentation requirements is not, and it does."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100043",
        decision="do_not_appeal",
        rule="Policy NC-11, documentation does not create coverage",
        reasoning=(
            "CARC 96 is a benefit exclusion, not a clinical determination. "
            "NC-11 states plainly that clinical documentation does not alter "
            "benefit design and that appeals consisting solely of additional "
            "clinical records against an exclusion are upheld. The file "
            "contains only clinical documentation and no benefit exception. An "
            "appeal on this record fails by the payer's own stated rule."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100044",
        decision="escalate",
        rule="No trusted definition for the denial code",
        reasoning=(
            "The denial code is absent from the curated table. A definition "
            "found by public search is unverified and cannot authorise an "
            "action. Escalation is not a failure to decide here, it is the "
            "correct decision: nobody in this system knows what the code means."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100045",
        decision="escalate",
        rule="Policy NC-11, conflicting prior authorization",
        reasoning=(
            "An approved authorization exists alongside a non-covered denial. "
            "NC-11 routes exactly this combination for manual review, because "
            "it means either the authorization was issued in error or the "
            "billed service differs from what was authorised, and neither is "
            "resolvable from the claim record. The authorization system "
            "confirms the second: PA-88213 authorises a different procedure "
            "than the one billed."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100046",
        decision="appeal",
        rule="Policy AU-07, authorization obtained but not submitted",
        reasoning=(
            "The authorization system of record confirms PA-77104 exists, is "
            "approved, covers the billed procedure, and the date of service "
            "falls inside its window. AU-07 states an appeal is accepted with "
            "the authorization number, the issue date, and evidence the "
            "authorised procedure, provider and date range match what was "
            "billed. Every one of those conditions is verified rather than "
            "asserted. Before day 13 this label would not have been defensible, "
            "because the only evidence was the claim notes saying so."
        ),
        certainty="rule",
    ),
]


def show() -> None:
    print("Golden set. Every label determined by a published rule.\n")
    for lab in GOLDEN:
        print(f"{lab.claim_id}   {lab.decision}")
        print(f"  rule: {lab.rule}")
        for line in _wrap(lab.reasoning, 68):
            print(f"  {line}")
        print()

    by_decision: dict[str, int] = {}
    for lab in GOLDEN:
        by_decision[lab.decision] = by_decision.get(lab.decision, 0) + 1
    print("=" * 70)
    print("  ".join(f"{k} {v}" for k, v in sorted(by_decision.items())))
    print(f"\n{len(GOLDEN)} labelled claims. This is a small set and it is the "
          f"whole set,\nso any accuracy figure computed from it carries that "
          f"sample size with it.")
    print("\nWhat is deliberately NOT asserted: whether a reviewer would agree "
          "\nwith the clinical merits. Where that is the question, the label "
          "is escalate.")


def record() -> None:
    store.init()
    for lab in GOLDEN:
        store.record_resolution(
            lab.claim_id,
            LABELLED_BY,
            lab.decision,
            f"{lab.rule}. {lab.reasoning}",
        )
    print(f"recorded {len(GOLDEN)} resolutions")
    print("\nNote: the agent now escalates when it proposes something that")
    print("contradicts one of these. That guardrail has existed since day 7")
    print("and has never fired outside a test, because the table was empty.")
    print("Evaluation runs should pass use_memory=False to avoid measuring")
    print("the agent against answers it has been shown.")


def clear() -> None:
    with store.connect() as conn:
        n = conn.execute(
            "DELETE FROM resolutions WHERE decided_by = ?",
            (LABELLED_BY,)).rowcount
        conn.commit()
    print(f"removed {n} resolution(s)")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    {"show": show, "record": record, "clear": clear}.get(cmd, show)()
