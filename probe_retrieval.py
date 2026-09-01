"""
Day 8: is retrieval actually load-bearing?

Written on day 6, run on day 8 — the day 6 attempts died on the daily quota
before producing an answer.

Real retrieval replaced the stub on day 6 and every case landed on the same
decision, with the same confidence, as it did with two hardcoded sentences.
That has two possible explanations and they are not distinguishable from the
run output:

  a) The corpus says roughly what the stub said, so the answer is genuinely
     the same.
  b) The model isn't using the retrieved text at all. It sees the category,
     recalls what that category means from training, and the retrieval is
     decoration.

This distinguishes them. It injects one policy passage that says something the
model would NOT assume, and that points the opposite way from its current
answer. Then it runs the same case.

  If the decision moves -> retrieval is load-bearing. The model reads.
  If nothing moves     -> the tool call is theatre, and every "the model
                          changed its mind after retrieving" result from this
                          project needs revisiting.

Baseline for CLM-100046: appeal at 0.85, recorded across three days and
two retrieval implementations. On 27 Aug the stub returned nothing at all for
this category; on 28 and 31 Aug real retrieval returned the correct passage,
then a better one. Same decision, same confidence, every time.

The injected document is removed afterwards unless you pass --keep.

    python probe_retrieval.py gemini-3.7-flash 3
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import retrieval as retrieval_module
from agent import (
    DenialAppealAgent,
    DenialCodeLookup,
    DenialRecord,
    ModelClient,
)

INJECTED = Path("policies") / "policy-authorization-amendment.md"

# Written to contradict the model's prior. Nothing in training says this payer
# stopped accepting these appeals, so a model that answers "appeal" after
# reading this is not reading it.
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

BASELINE = {"decision": "appeal", "confidence": 0.85}
PAUSE = 45


def confirm_retrievable() -> bool:
    """Costs nothing. If the passage doesn't come back from the retriever, the
    model never sees it and the whole test is meaningless."""
    retrieval_module._RETRIEVER = None
    r = retrieval_module.get_retriever()

    query = f"{CASE.payer_explanation} {CASE.documentation_summary}"
    extra = [
        "can this denial be appealed, what routes are available, "
        "are appeals accepted on this basis"
    ]
    text = r.retrieve_for_agent(query, category="authorization_missing",
                                k=3, extra_queries=extra)
    hits = [(c, sc) for c, sc in
            sorted({id(c): (c, sc) for q in [query] + extra
                    for c, sc in r.search(q, k=6, category="authorization_missing")}.values(),
                   key=lambda p: -p[1])]
    # show what retrieve_for_agent would actually hand over
    hits = []
    for line in text.split("\n"):
        if line.startswith("[") and "::" in line:
            hits.append(line)

    print(f"  backend: {r.backend.name}")
    print(f"  chunks:  {len(r.chunks)}")
    print("  what the agent will see for this case:")
    found = False
    for line in hits:
        mark = "  <-- injected" if "amendment" in line else ""
        print(f"    {line}{mark}")
        if "amendment" in line:
            found = True
    return found


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_MODEL", "gemini-3.7-flash")
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    keep = "--keep" in sys.argv

    if not Path("policies").exists():
        raise SystemExit("policies/ not found. Run: python build_corpus.py")

    print("Day 6 probe: is retrieval load-bearing?\n")
    print(f"model:    {model_name}")
    print(f"case:     {CASE.claim_id}")
    print(f"baseline: {BASELINE['decision']} at {BASELINE['confidence']}\n")

    INJECTED.write_text(INJECTED_TEXT, encoding="utf-8")
    print(f"injected: {INJECTED.name}")

    try:
        if not confirm_retrievable():
            print("\n  The injected passage did NOT surface in the top 3.")
            print("  The model would never see it, so the test proves nothing.")
            print("  Fix retrieval first. No API calls spent.")
            return

        print("\n  Injected passage is retrievable. Running.\n")

        results = []
        for i in range(1, runs + 1):
            a = DenialAppealAgent(
                code_lookup=DenialCodeLookup(),
                model=ModelClient(model=model_name),
                audit_log=False,
                # Day 8. CLM-100046 now has seven prior runs on record, all of
                # them visible to the model through memory. Leaving that on
                # would mean testing retrieval and history at the same time.
                use_memory=False,
            )
            try:
                state = a.run(CASE)
            except Exception as exc:
                print(f"  run {i}   FAILED  {type(exc).__name__}")
                results.append({"run": i, "error": type(exc).__name__})
                continue

            j = state.judgment
            rec = {
                "run": i,
                "tools_called": list(state.tools_called),
                "proposed_decision": j.proposed_decision if j else None,
                "confidence": j.confidence if j else None,
                "reasoning": j.reasoning_summary if j else None,
                "final_decision": state.decision.value if state.decision else None,
                "stop_reason": state.stop_reason,
            }
            results.append(rec)

            conf = rec["confidence"]
            print(f"  run {i}   proposed {rec['proposed_decision']}  "
                  f"conf {conf if conf is None else f'{conf:.2f}'}  "
                  f"final {rec['final_decision']}")
            if rec["reasoning"]:
                print(f"         {rec['reasoning'][:150]}")

            if i < runs:
                time.sleep(PAUSE)

        report(model_name, results)

    finally:
        if not keep and INJECTED.exists():
            INJECTED.unlink()
            retrieval_module._RETRIEVER = None
            print(f"\nremoved {INJECTED.name} (pass --keep to leave it in place)")


def report(model_name: str, results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("\nno runs completed")
        return

    moved = [r for r in ok if r["proposed_decision"] != BASELINE["decision"]]
    mentions = [
        r for r in ok
        if r["reasoning"] and any(
            w in r["reasoning"].lower()
            for w in ("corrected claim", "amendment", "no appeal", "au-09",
                      "discontinued", "administratively closed", "not accept")
        )
    ]

    print("\n" + "=" * 70)
    print(f"baseline:        {BASELINE['decision']} at {BASELINE['confidence']}")
    print(f"with amendment:  "
          f"{', '.join(str(r['proposed_decision']) for r in ok)}")
    print(f"decision moved:  {len(moved)} of {len(ok)}")
    print(f"reasoning references the new policy: {len(mentions)} of {len(ok)}")

    print()
    if len(moved) == len(ok):
        print("  Retrieval is load-bearing. One document changed the answer on")
        print("  every run. The model reads what comes back and acts on it.")
    elif moved:
        print("  Partly. The document moved some runs and not others, which is")
        print("  the same run-to-run instability from day 4 and 5, now visible")
        print("  in how much attention the model pays to retrieved text.")
    else:
        print("  Retrieval is NOT load-bearing on this case. A policy stating")
        print("  plainly that no appeal pathway exists did not change the")
        print("  answer once. The model is going on the category, not the text.")
        print("  Every earlier 'it changed its mind after retrieving' result")
        print("  needs revisiting.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"probe-retrieval-{model_name}-{stamp}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "case": CASE.claim_id,
                "baseline": BASELINE,
                "injected_document": INJECTED.name,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
