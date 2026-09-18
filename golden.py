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
    # ─────────────────────────────── Day 19: timely filing
    Label(
        claim_id="CLM-100047",
        decision="appeal",
        rule="Policy TF-02, accepted proof of timely submission",
        reasoning=(
            "TF-02 names a clearinghouse acceptance report identifying the "
            "claim within the filing window as accepted proof. The record has "
            "one, dated 12 days after the service. The denial is wrong on the "
            "payer's own stated criteria, and this is the only route TF-02 "
            "leaves open."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100048",
        decision="do_not_appeal",
        rule="Policy TF-02, what is not accepted",
        reasoning=(
            "No proof of submission of any kind. TF-02 states that clinical "
            "documentation is not accepted as proof of submission date, so the "
            "strong clinical record here is irrelevant to this denial. Without "
            "proof there is nothing to appeal with."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100049",
        decision="do_not_appeal",
        rule="Policy TF-02, what is not accepted",
        reasoning=(
            "A practice management screenshot is named explicitly in TF-02 as "
            "not accepted. The record looks like it contains proof and does "
            "not. An appeal on this basis fails by the stated rule."
        ),
        certainty="rule",
    ),

    # ─────────────────────────────── Day 19: duplicates
    Label(
        claim_id="CLM-100050",
        decision="do_not_appeal",
        rule="CARC 18, check whether the original paid",
        reasoning=(
            "The remittance shows the original claim adjudicated and paid in "
            "full. There is no underpayment and nothing to recover. Appealing "
            "a correctly identified duplicate is wasted effort."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100051",
        decision="escalate",
        rule="CARC 18, distinct services need a corrected claim",
        reasoning=(
            "The operative note documents two genuinely distinct procedures. "
            "The correct route is a corrected claim carrying a distinguishing "
            "modifier, not an appeal arguing the denial was wrong. The agent "
            "has no corrected-claim outcome, so the only honest action "
            "available to it is to hand this to a person."
        ),
        certainty="rule",
    ),

    # ─────────────────────────────── Day 19: missing information
    Label(
        claim_id="CLM-100052",
        decision="escalate",
        rule="Policy CI-03, corrected claim is the route",
        reasoning=(
            "CI-03 states that an appeal is not the appropriate route for a "
            "returned claim and will itself be returned. The missing element "
            "is known and the fix is a corrected claim. As with CLM-100051, "
            "the agent cannot file one, so the correct action is escalation "
            "rather than either appealing or closing."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100053",
        decision="escalate",
        rule="Policy CI-03, the missing element is unidentified",
        reasoning=(
            "No RARC and nothing in the record identifies which element is "
            "missing. Neither an appeal nor a corrected claim can be prepared "
            "without knowing what to supply. This needs a person to obtain the "
            "detail from the payer."
        ),
        certainty="rule",
    ),

    # ─────────────────────────────── Day 19: authorization variants
    Label(
        claim_id="CLM-100054",
        decision="escalate",
        rule="Policy AU-07, authorization status conflicts with the notes",
        reasoning=(
            "The authorization system records PA-55010 as revoked. The claim "
            "notes still reference it as obtained. Whether the revocation was "
            "correct, and whether the service was rendered before it took "
            "effect, is not resolvable from the claim record. AU-07 offers no "
            "route on a revoked authorization, and the conflict itself needs a "
            "person."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100055",
        decision="do_not_appeal",
        rule="Policy AU-07, the authorised window had closed",
        reasoning=(
            "PA-61200 is approved and covers the billed procedure, but its "
            "window ended 2026-03-10 and the service was 2026-05-02. AU-07 is "
            "explicit that the authorised date range must match what was "
            "billed. An authorization outside its window does not support the "
            "claim, and no appeal route follows from it."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100056",
        decision="do_not_appeal",
        rule="Policy AU-07, no authorization was issued",
        reasoning=(
            "The notes cite PA-99999. The authorization system of record has "
            "no such authorization. A reference in the claim notes is not "
            "evidence one was issued, so this falls under AU-07's 'no "
            "authorization obtained' branch, where retrospective authorization "
            "is rarely granted and the appeal usually fails."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100057",
        decision="escalate",
        rule="Policy AU-07, emergency exception needs review",
        reasoning=(
            "No authorization was requested and none exists, so the denial is "
            "correct on its face. But AU-07 allows retrospective authorization "
            "where the service met emergency criteria, and the record states "
            "the service was urgent without documenting the criteria. Whether "
            "it qualifies is a clinical judgment, which is the boundary where "
            "the correct answer is escalation."
        ),
        certainty="judgment",
    ),

    # ─────────────────────────────── Day 19: medical necessity variants
    Label(
        claim_id="CLM-100058",
        decision="do_not_appeal",
        rule="Policy MN-04, what a first-level appeal must contain",
        reasoning=(
            "MN-04 requires the diagnosis and severity, prior conservative "
            "management and its outcome, and the reasoning connecting them to "
            "the service. The record establishes none of these and states only "
            "that the procedure was performed. MN-04 says appeals that restate "
            "the service without establishing the indication are upheld at the "
            "initial denial."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100059",
        decision="do_not_appeal",
        rule="Policy MN-04, first-level appeals within 180 days",
        reasoning=(
            "The clinical record would support an appeal, and the window has "
            "closed. MN-04 requires a first-level appeal within 180 days of "
            "the remittance date; the remittance is dated 2025-09-15, roughly "
            "nine months before. A strong case filed too late is still not a "
            "case."
        ),
        certainty="rule",
    ),

    # ─────────────────────────────── Day 19: other categories
    Label(
        claim_id="CLM-100060",
        decision="do_not_appeal",
        rule="CARC 27, an eligibility dispute needs evidence",
        reasoning=(
            "Coverage terminated before the date of service and the record "
            "contains no eligibility verification, plan letter or evidence of "
            "retroactive reinstatement. This is resolved with evidence rather "
            "than argument, and there is none. The balance is a coordination "
            "of benefits question, not an appeal."
        ),
        certainty="rule",
    ),
    Label(
        claim_id="CLM-100061",
        decision="escalate",
        rule="CARC 109, rebill the correct payer",
        reasoning=(
            "The claim went to the wrong payer. The route is to identify the "
            "correct one and rebill, not to appeal. The agent cannot rebill, "
            "and the timely filing clock on the correct payer has been running "
            "since the date of service, which makes this time-sensitive. A "
            "person has to act."
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
