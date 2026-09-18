"""
The claim set. One definition per claim, imported everywhere.

Day 15. CLM-100046 existed in two files with different fields. `run_cases.py`
had no procedure code or date of service; `calibrate_tools.py` had both. The
authorization check uses those fields, so the same claim ID produced different
evidence depending on which file the test imported from.

That cost a measurement. Day 14's memory experiment ran the version with full
fields and found the agent appealed on 42% of runs without memory. Day 15's
evaluation ran the version without them and got 5 of 5. Two numbers about the
same claim ID, not comparable, and nothing in either file said so.

Second time a duplicated definition has broken a comparison. The first was the
memory condition drifting between sessions because the history kept
accumulating. Both are the same failure: a thing that looks fixed and isn't.

So: one file. `run_cases`, `calibrate_tools`, `evaluate` and the probes all
import from here.

Day 19 grew the set from five to twenty.

Five was not a dataset. Each claim was worth twenty percentage points, so "88%
correct" meant "one claim is unreliable" and little else. Three of the five
were trivially stable and two carried every interesting result.

It also left holes. Rules existed in the guardrail layer for timely filing,
duplicate claims and missing information that had never run against a claim,
and policy documents sat in the corpus that nothing ever retrieved.

What the fifteen new claims are for:

  the boring middle     most denials are not dramatic. A duplicate that really
                        is a duplicate. A missing NPI. If those are not easy,
                        that matters more than the hard cases do.
  every category        timely filing, duplicate, missing information,
                        coverage terminated, wrong payer
  the same trap again   several claims where the notes assert something the
                        system of record contradicts, because that is the
                        failure this project keeps finding
  route, not verdict    some denials are fixed with a corrected claim rather
                        than an appeal. The agent has no corrected-claim
                        outcome, so the right answer is escalate, and a set
                        that never tests it hides the gap.

    python cases.py        show every case and what the auth system says
"""

from __future__ import annotations

from agent import DenialRecord

# Meets the three things Policy MN-04 requires of a first-level appeal: the
# diagnosis and severity, prior conservative management and its outcome, and
# the reasoning connecting them to the service billed.
STRONG_DOCUMENTATION = (
    "Chart notes document persistent symptoms over four months, failure of "
    "prior conservative management including physical therapy, and the "
    "clinical indication for the procedure billed."
)


# Deliberately thin. Names the service and nothing else, so a claim carrying
# this cannot support an appeal on the merits.
THIN_DOCUMENTATION = (
    "Procedure performed as scheduled. No additional clinical detail on file."
)


def _claim(claim_id, patient_id, amount, carc, rarc, explanation,
           documentation, procedure_code=None, date_of_service="2026-06-20"):
    """Day 19. The original five are spelled out in full above for history.
    Fifteen more would be unreadable that way."""
    return DenialRecord(
        claim_id=claim_id,
        patient_id=patient_id,
        payer="Synthetic Health Plan",
        amount=amount,
        carc=carc,
        rarc=rarc,
        payer_explanation=explanation,
        documentation_summary=documentation,
        procedure_code=procedure_code,
        date_of_service=date_of_service,
    )


CLM_100042 = DenialRecord(
    claim_id="CLM-100042",
    patient_id="SYNTH-001",
    payer="Synthetic Health Plan",
    amount=1840.00,
    carc="50",
    rarc=None,
    payer_explanation="The payer states that the service was not medically necessary.",
    documentation_summary=STRONG_DOCUMENTATION,
    procedure_code="29826",
    date_of_service="2026-06-20",
)

CLM_100043 = DenialRecord(
    claim_id="CLM-100043",
    patient_id="SYNTH-002",
    payer="Synthetic Health Plan",
    amount=920.00,
    carc="96",
    rarc="N130",
    payer_explanation="This charge is not covered under the member's benefit plan.",
    documentation_summary=STRONG_DOCUMENTATION,
    procedure_code="0101T",
    date_of_service="2026-06-20",
)

CLM_100044 = DenialRecord(
    claim_id="CLM-100044",
    patient_id="SYNTH-003",
    payer="Synthetic Health Plan",
    amount=310.00,
    carc="99",
    rarc=None,
    payer_explanation="Denied. See remittance advice for details.",
    documentation_summary=STRONG_DOCUMENTATION,
    procedure_code="29826",
    date_of_service="2026-06-20",
)

# The claim notes assert the authorization covers "this exact procedure code".
# The authorization system says PA-88213 authorises 64483, and 29827 was
# billed. The notes are wrong, and only the system of record can show it.
CLM_100045 = DenialRecord(
    claim_id="CLM-100045",
    patient_id="SYNTH-004",
    payer="Synthetic Health Plan",
    amount=2450.00,
    carc="96",
    rarc="N130",
    payer_explanation="This service is not covered under the member's benefit plan.",
    documentation_summary=(
        "Prior authorization reference PA-88213 was approved by the payer on "
        "2026-06-02 for this exact procedure code. The member's benefit summary "
        "lists the service as covered when medically necessary. The treating "
        "clinician documented the indication, and the payer's own approval "
        "letter is on file."
    ),
    procedure_code="29827",
    date_of_service="2026-06-20",
)

# PA-77104 checks out completely: exists, approved, covers 29827, and the date
# of service falls inside its window.
CLM_100046 = DenialRecord(
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
    procedure_code="29827",
    date_of_service="2026-06-20",
)


# ═══════════════════════════════════════ Day 19: timely filing

# TF-02 accepts a clearinghouse acceptance report as proof of timely
# submission. The record has one, so the denial is wrong on its own terms.
CLM_100047 = _claim(
    "CLM-100047", "SYNTH-006", 640.00, "29", None,
    "The time limit for filing this claim has expired.",
    "Clearinghouse acceptance report dated 2026-07-02 identifies this claim "
    "as accepted and forwarded, 12 days after the date of service and inside "
    "the 180-day filing window.",
    procedure_code="99213", date_of_service="2026-06-20")

# No proof of any kind. TF-02 says clinical records are irrelevant to a filing
# denial, so the strong documentation here is a distractor.
CLM_100048 = _claim(
    "CLM-100048", "SYNTH-010", 415.00, "29", None,
    "The time limit for filing this claim has expired.",
    STRONG_DOCUMENTATION, procedure_code="99214",
    date_of_service="2025-11-05")

# Proof exists but only as an internal screenshot, which TF-02 names as not
# accepted. Looks like evidence, is not.
CLM_100049 = _claim(
    "CLM-100049", "SYNTH-011", 880.00, "29", None,
    "The time limit for filing this claim has expired.",
    "Billing staff captured a screenshot of the practice management system "
    "showing the claim marked as submitted on 2026-02-14. No clearinghouse "
    "report or prior remittance is on file.",
    procedure_code="99215", date_of_service="2026-02-01")


# ═══════════════════════════════════════ Day 19: duplicates

# A real duplicate. The original paid. Nothing to appeal.
CLM_100050 = _claim(
    "CLM-100050", "SYNTH-012", 220.00, "18", None,
    "Exact duplicate claim or service.",
    "Remittance advice dated 2026-06-28 shows claim CLM-100050A for the same "
    "date of service and procedure was adjudicated and paid in full.",
    procedure_code="99213")

# Two genuinely distinct procedures on one day. The route is a corrected claim
# with a modifier, not an appeal, and the agent has no corrected-claim
# outcome, so this has to reach a human.
CLM_100051 = _claim(
    "CLM-100051", "SYNTH-013", 540.00, "18", None,
    "Exact duplicate claim or service.",
    "Operative note documents two separate procedures performed at different "
    "anatomical sites during the same encounter. The second was submitted "
    "without a distinguishing modifier.",
    procedure_code="29881")


# ═══════════════════════════════════════ Day 19: missing information

# CI-03 is explicit: supply the missing element and send a corrected claim.
# An appeal is the wrong route and gets returned.
CLM_100052 = _claim(
    "CLM-100052", "SYNTH-014", 305.00, "16", "M127",
    "Claim lacks information or contains submission errors.",
    "The rendering provider NPI was omitted from the original submission. The "
    "provider is credentialed with this payer and the NPI is on file.",
    procedure_code="99213")

# Same category, but nothing says what is actually missing. Nobody can correct
# a claim without knowing which element to correct.
CLM_100053 = _claim(
    "CLM-100053", "SYNTH-015", 760.00, "16", None,
    "Claim lacks information or contains submission errors.",
    THIN_DOCUMENTATION, procedure_code="20610")


# ═══════════════════════════════════════ Day 19: authorization variants

# PA-55010 was revoked after issue. The notes still cite it and nothing in the
# claim record would ever reveal the revocation.
CLM_100054 = _claim(
    "CLM-100054", "SYNTH-007", 3100.00, "197", None,
    "Precertification, authorization or notification was absent.",
    "Scheduling notes reference prior authorization PA-55010 obtained before "
    "the date of service.",
    procedure_code="27447", date_of_service="2026-05-20")

# PA-61200 expired before the date of service.
CLM_100055 = _claim(
    "CLM-100055", "SYNTH-008", 1420.00, "197", None,
    "Precertification, authorization or notification was absent.",
    "Prior authorization PA-61200 was approved for this member and this "
    "procedure.",
    procedure_code="43239", date_of_service="2026-05-02")

# The notes cite an authorization the system never issued.
CLM_100056 = _claim(
    "CLM-100056", "SYNTH-009", 990.00, "197", None,
    "Precertification, authorization or notification was absent.",
    "Prior authorization PA-99999 was obtained before the date of service and "
    "is referenced in the scheduling notes.",
    procedure_code="29826")

# Nothing claimed, nothing on file. The denial stands.
CLM_100057 = _claim(
    "CLM-100057", "SYNTH-016", 1180.00, "197", None,
    "Precertification, authorization or notification was absent.",
    "Service performed on an urgent basis. No prior authorization was "
    "requested and none is referenced in the record.",
    procedure_code="29827")


# ═══════════════════════════════════════ Day 19: medical necessity variants

# MN-04 upholds appeals that restate the service without establishing the
# indication, and that is all this record does.
CLM_100058 = _claim(
    "CLM-100058", "SYNTH-017", 1650.00, "50", None,
    "The payer states that the service was not medically necessary.",
    THIN_DOCUMENTATION, procedure_code="29826")

# Well documented, but the remittance is nine months old and MN-04 allows 180
# days for a first-level appeal. The route has closed.
CLM_100059 = _claim(
    "CLM-100059", "SYNTH-018", 2240.00, "50", None,
    "The payer states that the service was not medically necessary. "
    "Remittance issued 2025-09-15.",
    STRONG_DOCUMENTATION + " The remittance advice is dated 2025-09-15.",
    procedure_code="29826", date_of_service="2025-08-30")


# ═══════════════════════════════════════ Day 19: other categories

# An eligibility dispute, resolved with evidence rather than argument, and the
# record has none.
CLM_100060 = _claim(
    "CLM-100060", "SYNTH-019", 470.00, "27", None,
    "Expenses incurred after coverage terminated.",
    "No eligibility verification was captured on the date of service and no "
    "employer or plan letter is on file.",
    procedure_code="99213")

# Not an appeal at all: identify the correct payer and rebill.
CLM_100061 = _claim(
    "CLM-100061", "SYNTH-020", 1290.00, "109", None,
    "Claim not covered by this payer or contractor.",
    "Member enrolled in a Medicare Advantage plan effective 2026-01-01. The "
    "claim was submitted to traditional Medicare.",
    procedure_code="99214")


ALL = [
    CLM_100042, CLM_100043, CLM_100044, CLM_100045, CLM_100046,
    CLM_100047, CLM_100048, CLM_100049, CLM_100050, CLM_100051,
    CLM_100052, CLM_100053, CLM_100054, CLM_100055, CLM_100056,
    CLM_100057, CLM_100058, CLM_100059, CLM_100060, CLM_100061,
]

LABELS = {
    "CLM-100042": "medical necessity, well documented",
    "CLM-100043": "non-covered charge, well documented",
    "CLM-100044": "unmapped denial code",
    "CLM-100045": "non-covered charge, notes contradicted by the auth system",
    "CLM-100046": "authorization missing, verified authorization on file",
    "CLM-100047": "timely filing, clearinghouse proof on file",
    "CLM-100048": "timely filing, no proof of submission",
    "CLM-100049": "timely filing, proof TF-02 does not accept",
    "CLM-100050": "duplicate, and the original was paid",
    "CLM-100051": "duplicate, but genuinely two procedures",
    "CLM-100052": "missing information, the missing element is known",
    "CLM-100053": "missing information, nothing says what is missing",
    "CLM-100054": "authorization revoked after issue",
    "CLM-100055": "authorization expired before the service",
    "CLM-100056": "authorization the system never issued",
    "CLM-100057": "no authorization claimed and none on file",
    "CLM-100058": "medical necessity, thin documentation",
    "CLM-100059": "medical necessity, appeal window closed",
    "CLM-100060": "coverage terminated, no eligibility evidence",
    "CLM-100061": "wrong payer, needs rebilling not appealing",
}

BY_ID = {c.claim_id: c for c in ALL}


def label(claim_id: str) -> str:
    return LABELS.get(claim_id, claim_id)


if __name__ == "__main__":
    from agent import check_prior_authorization

    for case in ALL:
        print(f"{case.claim_id}   {LABELS[case.claim_id]}")
        print(f"  CARC {case.carc}   ${case.amount:,.2f}   "
              f"proc {case.procedure_code}   dos {case.date_of_service}")
        result = check_prior_authorization(case, None)
        first = result.split("\n")[0]
        print(f"  auth system: {first}")
        if "PROBLEMS FOUND" in result:
            for line in result.split("\n"):
                if line.startswith("- "):
                    print(f"    {line}")
        print()
