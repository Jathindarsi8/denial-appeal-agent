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


ALL = [CLM_100042, CLM_100043, CLM_100044, CLM_100045, CLM_100046]

LABELS = {
    "CLM-100042": "medical necessity, well documented",
    "CLM-100043": "non-covered charge, well documented",
    "CLM-100044": "unmapped denial code",
    "CLM-100045": "non-covered charge, notes contradicted by the auth system",
    "CLM-100046": "authorization missing, verified authorization on file",
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
