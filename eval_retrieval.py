"""
Retrieval evaluation.

Day 6 shipped a retriever with a bug that made it structurally unable to return
the one document that decided a case: the top three came back as three chunks
of one file. That was found by accident, because a document was planted and
then looked for. Accident is not a test strategy.

This is the fixed version of that check. Each query names the document that
must come back. Recall@k is then just: did it.

The labels are legitimate rather than invented. The corpus was written for this
project, so which document answers which question is known, not guessed. Where
a query could reasonably be answered by more than one document, all acceptable
documents are listed and a hit on any counts.

Costs nothing to run. No API calls, no quota.

    python eval_retrieval.py            # summary
    python eval_retrieval.py --verbose  # every query, with what came back
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from retrieval import get_retriever

# The second query the agent sends alongside the claim text. Included here so
# the evaluation measures what the agent actually does, not a simpler version
# of it.
DECISION_QUERY = (
    "can this denial be appealed, what routes are available, "
    "are appeals accepted on this basis"
)


@dataclass
class Case:
    query: str
    category: str | None
    expect: set[str]      # any one of these counts as a hit
    note: str = ""


CASES: list[Case] = [
    # ---------------------------------------------------- authorization
    Case("authorization was obtained but left off the claim form",
         "authorization_missing",
         {"policy-authorization.md", "carc-197-authorization-missing.md"},
         "the core scenario"),
    Case("no precertification was ever requested for this service",
         "authorization_missing",
         {"carc-197-authorization-missing.md", "policy-authorization.md"},
         "opposite situation, same denial code"),
    Case("can we still appeal if the authorization number is missing",
         "authorization_missing",
         {"policy-authorization.md", "carc-197-authorization-missing.md"}),
    Case("the authorized procedure code does not match what was billed",
         "authorization_missing",
         {"policy-authorization.md"},
         "partial match voids the authorization"),
    Case("emergency service rendered without prior approval",
         "authorization_missing",
         {"policy-authorization.md", "carc-197-authorization-missing.md"}),

    # ---------------------------------------------------- non-covered
    Case("service excluded under the member's benefit plan",
         "noncovered_charge",
         {"carc-96-non-covered.md", "policy-non-covered.md"}),
    Case("we have full clinical documentation, does that make it covered",
         "noncovered_charge",
         {"policy-non-covered.md", "carc-96-non-covered.md"},
         "documentation does not create coverage"),
    Case("how do we request an exception for an excluded service",
         "noncovered_charge",
         {"policy-non-covered.md"},
         "benefit exception route"),
    Case("approved prior authorization on a claim denied as non-covered",
         "noncovered_charge",
         {"policy-non-covered.md", "carc-96-non-covered.md"},
         "the contradiction case"),
    Case("was this billed under the wrong procedure code",
         "noncovered_charge",
         {"policy-non-covered.md"},
         "coding correction route"),

    # ---------------------------------------------------- medical necessity
    Case("denied as not medically necessary",
         "medical_necessity",
         {"carc-50-medical-necessity.md", "policy-medical-necessity.md"}),
    Case("what does a first level appeal need to contain",
         "medical_necessity",
         {"policy-medical-necessity.md"}),
    Case("patient failed conservative treatment before this procedure",
         "medical_necessity",
         {"policy-medical-necessity.md", "carc-50-medical-necessity.md"}),
    Case("can we request a peer to peer review instead of writing an appeal",
         "medical_necessity",
         {"policy-medical-necessity.md"},
         "specific route buried in a subsection"),
    Case("how long do we have to file a medical necessity appeal",
         "medical_necessity",
         {"policy-medical-necessity.md"}),

    # ---------------------------------------------------- timely filing
    Case("claim filed after the deadline",
         "timely_filing",
         {"carc-29-timely-filing.md", "policy-timely-filing.md"},
         "the query TF-IDF could not match"),
    Case("we submitted on time but they say we did not",
         "timely_filing",
         {"policy-timely-filing.md", "carc-29-timely-filing.md"},
         "proof of submission"),
    Case("does a clearinghouse report count as proof",
         "timely_filing",
         {"policy-timely-filing.md"}),
    Case("does sending a corrected claim reset the filing clock",
         "timely_filing",
         {"policy-timely-filing.md"}),

    # ---------------------------------------------------- missing information
    Case("claim rejected for missing information",
         "missing_or_invalid_information",
         {"carc-16-missing-information.md", "policy-missing-information.md"}),
    Case("should we appeal or send a corrected claim",
         "missing_or_invalid_information",
         {"policy-missing-information.md", "carc-16-missing-information.md"}),
    Case("which fields are commonly missing on returned claims",
         "missing_or_invalid_information",
         {"policy-missing-information.md", "carc-16-missing-information.md"}),

    # ---------------------------------------------------- duplicate / other
    Case("payer says this is a duplicate of an earlier claim",
         "duplicate_claim",
         {"carc-18-duplicate.md"}),
    Case("two identical procedures performed on the same day",
         "duplicate_claim",
         {"carc-18-duplicate.md"},
         "legitimately distinct, needs a modifier"),
    Case("member coverage had already ended on the date of service",
         "other",
         {"carc-27-coverage-terminated.md"}),
    Case("this was sent to the wrong payer",
         "other",
         {"carc-109-wrong-payer.md"}),

    # ---------------------------------------------------- cross-cutting
    Case("what has to be in an appeal for it to be reviewed at all",
         None,
         {"policy-appeal-format.md"},
         "general guidance, no category filter"),
    Case("appeal was closed without anyone reading it",
         None,
         {"policy-appeal-format.md"},
         "no basis stated"),
    Case("what happens after a first level appeal is denied",
         None,
         {"policy-appeal-format.md"},
         "escalation levels"),

    # ---------------------------------------------------- the trap
    Case("clinical records proving the treatment was necessary",
         "noncovered_charge",
         {"policy-non-covered.md", "carc-96-non-covered.md"},
         "TRAP: sounds like medical necessity, but the category is "
         "non-covered. Must not return the medical necessity policy."),
]


def evaluate(k: int = 3, use_decision_query: bool = True,
             verbose: bool = False) -> dict:
    r = get_retriever()

    hits_at_1 = 0
    hits_at_k = 0
    misses: list[tuple[Case, list[str]]] = []

    for case in CASES:
        extra = [DECISION_QUERY] if use_decision_query else None
        text = r.retrieve_for_agent(case.query, category=case.category,
                                    k=k, extra_queries=extra, max_per_doc=1)

        returned = []
        for line in text.split("\n"):
            if line.startswith("[") and "::" in line:
                returned.append(line[1:].split(" ::")[0])

        if returned and returned[0] in case.expect:
            hits_at_1 += 1
        if any(d in case.expect for d in returned):
            hits_at_k += 1
        else:
            misses.append((case, returned))

        if verbose:
            mark = "ok  " if any(d in case.expect for d in returned) else "MISS"
            print(f"{mark}  {case.query}")
            print(f"        expected any of: {', '.join(sorted(case.expect))}")
            print(f"        got:             {', '.join(returned) or '(nothing)'}")
            if case.note:
                print(f"        {case.note}")
            print()

    n = len(CASES)
    return {
        "n": n,
        "recall_at_1": hits_at_1 / n,
        "recall_at_k": hits_at_k / n,
        "k": k,
        "misses": misses,
    }


def main() -> None:
    verbose = "--verbose" in sys.argv
    k = 3

    r = get_retriever()
    print(f"backend: {r.backend.name}")
    print(f"corpus:  {len(r.chunks)} chunks from "
          f"{len(set(c.doc for c in r.chunks))} documents")
    print(f"queries: {len(CASES)}\n")

    result = evaluate(k=k, verbose=verbose)

    print("=" * 70)
    print(f"recall@1  {result['recall_at_1']:.0%}   "
          f"the right document ranked first")
    print(f"recall@{k}  {result['recall_at_k']:.0%}   "
          f"the right document came back at all")

    if result["misses"]:
        print(f"\n{len(result['misses'])} queries missed entirely:\n")
        for case, returned in result["misses"]:
            print(f"  {case.query}")
            print(f"    expected any of: {', '.join(sorted(case.expect))}")
            print(f"    got:             {', '.join(returned) or '(nothing)'}")
            if case.note:
                print(f"    note: {case.note}")
            print()
    else:
        print("\nno misses")

    # The comparison that justified installing torch, run as a measurement
    # rather than an anecdote.
    print("=" * 70)
    print("without the second decision-oriented query:")
    single = evaluate(k=k, use_decision_query=False)
    print(f"  recall@1  {single['recall_at_1']:.0%}   "
          f"recall@{k}  {single['recall_at_k']:.0%}")
    delta = result["recall_at_k"] - single["recall_at_k"]
    if delta > 0:
        print(f"  the second query is worth {delta:+.0%} recall@{k}")
    elif delta < 0:
        print(f"  the second query COSTS {delta:.0%} recall@{k} — it is "
              f"crowding out better hits")
    else:
        print(f"  the second query makes no difference to recall@{k} here")


if __name__ == "__main__":
    main()
