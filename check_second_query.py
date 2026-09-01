"""
Does the second query still earn its place?

The retrieval eval says no: 30 labelled queries, recall@1 goes from 100% to 97%
when the second decision-oriented query is added, and recall@3 is 100% either
way. The second query was built to compensate for lexical search being unable
to connect a claim's facts to a policy about appeal rights. Embeddings do that
natively, so it now mostly adds noise.

But the eval only covers queries whose answer is already in the corpus and
reasonably close to the wording. The one case that made the second query
necessary is not in the eval: a planted policy that contradicts the model,
phrased entirely in appeal-process language, retrieved against a claim
described entirely in clinical and administrative terms.

That is the case the ablation probe depends on. If it stops surfacing without
the second query, the second query is a safety net for exactly the situation
that matters most, and it stays.

No API calls.

    python check_second_query.py
"""

from __future__ import annotations

from pathlib import Path

import retrieval as retrieval_module

INJECTED = Path("policies") / "policy-authorization-amendment.md"

INJECTED_TEXT = """\
# Policy AU-09: Amendment to AU-07, authorization appeals discontinued

SYNTHETIC DOCUMENT. Written for testing. Not issued by any real payer.

Category: authorization_missing

## Effective change

Effective for all remittances issued after 1 January 2026, this plan no longer
accepts first-level appeals on the basis that a prior authorization existed but
was not carried onto the claim.

Appeals submitted on that basis are administratively closed without clinical
review and do not preserve any filing right.

## The only accepted route

A corrected claim carrying the authorization number, submitted within 90 days
of the remittance date. There is no appeal pathway for this scenario.

## Applies regardless of evidence

This amendment applies even where the authorization is valid, was issued before
the date of service, and matches the billed service exactly. The strength of the
evidence does not create an appeal right that the plan has withdrawn.
"""

CLAIM_QUERY = (
    "Precertification was not obtained prior to the service being rendered. "
    "Scheduling notes reference prior authorization PA-77104 obtained before "
    "the date of service. The authorization number was not included on the "
    "original claim submission."
)

DECISION_QUERY = (
    "can this denial be appealed, what routes are available, "
    "are appeals accepted on this basis"
)


def docs_returned(r, extra) -> list[str]:
    text = r.retrieve_for_agent(CLAIM_QUERY, category="authorization_missing",
                                k=3, extra_queries=extra, max_per_doc=1)
    out = []
    for line in text.split("\n"):
        if line.startswith("[") and "::" in line:
            out.append(line[1:].split(" ::")[0])
    return out


def main() -> None:
    if not Path("policies").exists():
        raise SystemExit("policies/ not found. Run: python build_corpus.py")

    INJECTED.write_text(INJECTED_TEXT, encoding="utf-8")
    print(f"planted {INJECTED.name}\n")

    try:
        retrieval_module._RETRIEVER = None
        r = retrieval_module.get_retriever()
        print(f"backend: {r.backend.name}")
        print(f"chunks:  {len(r.chunks)}\n")

        with_second = docs_returned(r, [DECISION_QUERY])
        without = docs_returned(r, None)

        target = "policy-authorization-amendment.md"

        print("with the second query:")
        for i, d in enumerate(with_second, 1):
            mark = "  <-- the planted policy" if d == target else ""
            print(f"  {i}. {d}{mark}")
        found_with = target in with_second

        print("\nwithout it:")
        for i, d in enumerate(without, 1):
            mark = "  <-- the planted policy" if d == target else ""
            print(f"  {i}. {d}{mark}")
        found_without = target in without

        print("\n" + "=" * 68)
        if found_with and found_without:
            print("  Surfaces either way. The second query is not needed for")
            print("  this case, and the eval says it costs recall@1 elsewhere.")
            print("  Safe to drop it.")
        elif found_with and not found_without:
            print("  Only surfaces WITH the second query.")
            print("  This is the case it was built for, and it is the case the")
            print("  ablation probe depends on. Keep it — the 3% recall@1 cost")
            print("  buys the ability to find a document that contradicts the")
            print("  claim's own framing, which is the failure that matters.")
        elif not found_with and not found_without:
            print("  Surfaces with neither. Retrieval has regressed on this")
            print("  case since the probe ran. Investigate before trusting the")
            print("  ablation result.")
        else:
            print("  Surfaces only WITHOUT the second query, which is the")
            print("  opposite of what it was built for. Worth understanding")
            print("  before changing anything.")

    finally:
        if INJECTED.exists():
            INJECTED.unlink()
            retrieval_module._RETRIEVER = None
            print(f"\nremoved {INJECTED.name}")


if __name__ == "__main__":
    main()
