"""
Day 6: build the policy corpus.

Until now retrieve_policy() was an if-statement returning a hardcoded string
with an invented policy number. Every finding so far rests on an agent
"retrieving" from a stub, which is the largest remaining hole in this project.

This writes a corpus of documents to policies/ for the retriever to index.

Two kinds of document:

  carc-*.md      CARC code reference. The code numbers and their meanings are
                 the real X12 standard codes, which are public. Verify against
                 x12.org/codes/claim-adjustment-reason-codes before trusting
                 any of it in a real setting.

  policy-*.md    Payer policy statements. These are SYNTHETIC. They are written
                 to read like real payer policy so the retrieval problem is
                 realistic, but no payer published them and the policy numbers
                 are invented. Never quote them as if they were real.

    python build_corpus.py
"""

from pathlib import Path

OUT = Path("policies")

DOCS: dict[str, str] = {}

# ─────────────────────────────────────────────────────── CARC reference

DOCS["carc-16-missing-information.md"] = """\
# CARC 16 — Claim/service lacks information or has submission/billing errors

Category: missing_or_invalid_information

The payer could not adjudicate the claim as submitted because required data is
missing, incomplete, or internally inconsistent. This is a completeness problem,
not a coverage decision. The payer has not said the service is uncovered.

Commonly paired with RARCs identifying the specific missing element: modifier,
units, place of service, referring provider NPI, or a required attachment.

Appeal posture: usually a corrected claim rather than an appeal. Identify the
missing element from the RARC, supply it, and resubmit. An appeal that does not
supply the missing element will be denied again for the same reason.
"""

DOCS["carc-29-timely-filing.md"] = """\
# CARC 29 — The time limit for filing has expired

Category: timely_filing

The claim reached the payer after the filing deadline in the provider contract
or plan document. Filing limits vary by payer and by contract, commonly between
90 and 365 days from the date of service.

Appeal posture: an appeal succeeds only with proof of timely original
submission, such as a clearinghouse acceptance report, an earlier remittance
showing the claim in process, or documentation of a payer system outage.
Clinical documentation is irrelevant to this denial. Without proof of timely
submission there is nothing to appeal.
"""

DOCS["carc-50-medical-necessity.md"] = """\
# CARC 50 — Not deemed a medical necessity by the payer

Category: medical_necessity

The payer accepts that the service is a covered benefit but has determined it
was not medically necessary for this patient on this date of service, based on
the information available to it at adjudication.

This denial turns on clinical facts, which means it is genuinely appealable when
the record supports the indication.

Appeal posture: the strongest appeals establish the diagnosis and severity, what
was tried before and why it failed or was inappropriate, and why this specific
service was indicated. Cite the payer's own medical policy criteria where they
are met. Attach the relevant chart notes rather than summarising them.
"""

DOCS["carc-96-non-covered.md"] = """\
# CARC 96 — Non-covered charge

Category: noncovered_charge

The service is excluded by the member's benefit plan. This is a benefit design
decision, not a clinical judgment about this patient.

The distinction matters and is frequently confused with CARC 50. CARC 50 says
"covered, but not necessary here." CARC 96 says "not part of this plan at all."

Appeal posture: clinical documentation does not change plan design. Submitting
more chart notes against a CARC 96 will not succeed. The available routes are a
benefit exception request, an appeal arguing the service was miscoded and is
actually a covered benefit, or a demonstration that the exclusion does not apply
to this member's specific plan document.

A CARC 96 with an approved prior authorization on file is a contradiction that
requires a human. It usually means either the authorization was issued in error,
or the claim was coded differently from what was authorised. Neither is
resolvable from the claim record alone.
"""

DOCS["carc-197-authorization-missing.md"] = """\
# CARC 197 — Precertification, authorization, or notification absent

Category: authorization_missing

The payer required prior authorization for this service and has no record of one
associated with the claim.

Two distinct situations produce this code, and they have opposite outcomes.

No authorization was ever obtained. The service was rendered without meeting a
contractual requirement. Retrospective authorization is rarely granted, and the
appeal usually fails absent a documented emergency exception.

An authorization existed but was not carried onto the claim. This is an
administrative failure, not a coverage failure. The authorization number was
omitted from the claim form, or the claim was submitted under a different
procedure code, provider, or date range than the one authorized.

Appeal posture: establish which situation applies before anything else. Where an
authorization exists, the appeal supplies the authorization number and evidence
it was issued before the date of service, and shows that the authorized service
matches what was billed. Where no authorization exists, review whether an
emergency or retro-authorization pathway applies before appealing.
"""

DOCS["carc-18-duplicate.md"] = """\
# CARC 18 — Exact duplicate claim or service

Category: duplicate_claim

The payer has matched this claim to one already received. Duplicates arise from
resubmission before the original adjudicated, clearinghouse resends, or the same
service legitimately performed twice on one day.

Appeal posture: check whether the original was paid before doing anything. If it
was, there is nothing to appeal. If the services are genuinely distinct, the
correct route is a corrected claim with an appropriate modifier distinguishing
them, not an appeal arguing the denial was wrong.
"""

DOCS["carc-27-coverage-terminated.md"] = """\
# CARC 27 — Expenses incurred after coverage terminated

Category: other

The payer's eligibility records show the member's coverage ended before the date
of service.

Appeal posture: this is an eligibility dispute, resolved with evidence rather
than argument. Useful evidence includes an eligibility verification response
captured on the date of service, an employer or plan letter confirming coverage,
or evidence of retroactive reinstatement. Where coverage genuinely ended, the
balance may be patient responsibility or billable to a different payer, and the
question is coordination of benefits rather than appeal.
"""

DOCS["carc-109-wrong-payer.md"] = """\
# CARC 109 — Claim not covered by this payer or contractor

Category: other

The claim was sent to a payer that does not hold this risk. Common causes are a
member changing plans, a Medicare Advantage member billed to traditional
Medicare, or the wrong entity in a delegated arrangement.

Appeal posture: usually not an appeal. Identify the correct payer and rebill.
Watch the timely filing clock on the correct payer, which has been running since
the date of service, not since this denial.
"""

# ─────────────────────────────────────────────────────── payer policy (synthetic)

DOCS["policy-medical-necessity.md"] = """\
# Policy MN-04: Medical necessity review and appeal

SYNTHETIC DOCUMENT. Written for testing. Not issued by any real payer.

Category: medical_necessity

## Standard applied

A service is medically necessary when it is consistent with the diagnosis,
consistent with generally accepted standards of practice, not primarily for the
convenience of the patient or provider, and provided at the most appropriate
level of care.

## What a first-level appeal must contain

The submitted record must establish the working diagnosis and clinical severity
at the time of the decision, prior conservative management attempted and its
outcome, and the clinical reasoning connecting the two to the service billed.

Appeals that restate the service without establishing the indication are upheld
at the initial denial.

## Timeframes

First-level appeals must be received within 180 days of the remittance date.
The plan responds within 30 days for pre-service and 60 days for post-service.

## Peer-to-peer

A peer-to-peer review may be requested within 14 days of the denial and is
frequently faster than a written appeal where the record is already complete.
"""

DOCS["policy-non-covered.md"] = """\
# Policy NC-11: Non-covered services and benefit exclusions

SYNTHETIC DOCUMENT. Written for testing. Not issued by any real payer.

Category: noncovered_charge

## Scope

This policy governs services excluded from coverage under the member's benefit
plan, regardless of medical necessity.

## Documentation does not create coverage

Clinical documentation does not alter benefit design. A service excluded by the
plan document remains excluded however well the clinical indication is
established. Appeals that consist solely of additional clinical records against
an exclusion are upheld.

## Available routes

A benefit exception is a separate written determination requiring plan sponsor
involvement and is not processed through the standard appeal queue.

A coding correction is appropriate where the service billed was mischaracterised
and the correctly coded service is a covered benefit.

A plan document review is appropriate where the member's specific plan does not
contain the exclusion applied.

## Conflicting prior authorization

Where an approved prior authorization exists for a service later denied as
non-covered, the claim must be routed for manual review. This combination
indicates either an authorization issued in error or a mismatch between the
authorized and billed service, and cannot be resolved from the claim record.
"""

DOCS["policy-authorization.md"] = """\
# Policy AU-07: Prior authorization requirements and post-service review

SYNTHETIC DOCUMENT. Written for testing. Not issued by any real payer.

Category: authorization_missing

## Requirement

Services on the plan's authorization list require an approved authorization
before the service is rendered. The authorization number must appear on the
claim.

## Authorization obtained but not submitted

Where a valid authorization was issued before the date of service but was not
included on the claim, the correct route is a corrected claim carrying the
authorization number. Where the claim has already been denied, an appeal is
accepted with the authorization number, the issue date, and evidence that the
authorized procedure, provider, and date range match what was billed.

A mismatch between the authorized service and the billed service voids the
authorization for this purpose. The plan treats a partially matching
authorization as no authorization.

## No authorization obtained

Retrospective authorization is granted only where the service met emergency
criteria, or where the member's eligibility could not reasonably have been
determined before the service. Documentation of the emergency or the eligibility
failure is required.

## Timeframes

Corrected claims within 90 days of the remittance date. Appeals within 180 days.
"""

DOCS["policy-timely-filing.md"] = """\
# Policy TF-02: Filing deadlines and proof of timely submission

SYNTHETIC DOCUMENT. Written for testing. Not issued by any real payer.

Category: timely_filing

## Deadline

Claims must be received within 180 days of the date of service unless the
provider agreement specifies otherwise. The provider agreement controls where
the two conflict.

## Accepted proof of timely submission

A clearinghouse acceptance report identifying the claim and showing a date
within the filing window. A prior remittance advice showing the claim in
process. Written confirmation of a payer system outage covering the period.

## Not accepted

Internal practice management system screenshots, statements of office practice,
and clinical documentation are not accepted as proof of submission date.

## Effect of a corrected claim

Submitting a corrected claim does not reset the filing clock. The original
submission date governs.
"""

DOCS["policy-missing-information.md"] = """\
# Policy CI-03: Incomplete claims and corrected claim submission

SYNTHETIC DOCUMENT. Written for testing. Not issued by any real payer.

Category: missing_or_invalid_information

## Handling

A claim that cannot be adjudicated for missing or invalid data is returned
rather than denied on the merits. No coverage determination has been made.

## Correct route

Supply the missing element and submit a corrected claim. The accompanying RARC
identifies which element is missing. An appeal is not the appropriate route and
will be returned.

## Common missing elements

Rendering provider identifier, procedure or diagnosis code specificity, required
modifier, units, place of service, and required attachments for services on the
attachment list.

## Filing clock

The original submission date is preserved where the corrected claim is received
within 90 days of the return.
"""

DOCS["policy-appeal-format.md"] = """\
# Policy AP-01: Appeal submission requirements

SYNTHETIC DOCUMENT. Written for testing. Not issued by any real payer.

Category: any

## Required elements

Every appeal must identify the member, the claim number, the dates of service in
dispute, the specific adjustment being appealed, and the basis for the appeal.

An appeal that does not state a basis is administratively closed without review.

## Basis must match the denial

The basis for the appeal must respond to the reason given. An appeal arguing
medical necessity against a benefit exclusion, or arguing clinical facts against
a timely filing denial, is upheld without clinical review.

## Levels

First level is a written reconsideration. Second level is an independent review
by a reviewer not involved in the first determination. External review is
available after internal levels are exhausted, subject to plan and jurisdiction.

## Representation

An appeal filed by a provider on the member's behalf may require a signed
authorization of representation depending on plan type.
"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, body in DOCS.items():
        (OUT / name).write_text(body, encoding="utf-8")
    print(f"wrote {len(DOCS)} documents to {OUT}/")
    total = sum(len(b.split()) for b in DOCS.values())
    print(f"roughly {total} words of corpus")
    print("\ncarc-*.md    real X12 code meanings, verify at x12.org before real use")
    print("policy-*.md  synthetic, written for testing, no payer issued them")


if __name__ == "__main__":
    main()
