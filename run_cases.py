"""
Day 2 verification.

Same four cases as Day 1, plus one designed to make the model reach for a tool.
What's new: the trace now shows which tools the model chose to call, and why.
"""

from agent import (
    DenialAppealAgent,
    DenialCodeLookup,
    DenialRecord,
    ModelClient,
)

STRONG_DOCUMENTATION = (
    "The treating clinician documented persistent symptoms, prior conservative "
    "treatment failure, and the clinical indication for the ordered service."
)

CASES = [
    (
        "medical necessity, well documented",
        DenialRecord(
            claim_id="CLM-100042",
            patient_id="SYNTH-001",
            payer="Synthetic Health Plan",
            amount=1840.00,
            carc="50",
            rarc=None,
            payer_explanation="The payer states that the service was not medically necessary.",
            documentation_summary=STRONG_DOCUMENTATION,
        ),
        "appeal",
    ),
    (
        "non-covered charge, well documented",
        DenialRecord(
            claim_id="CLM-100043",
            patient_id="SYNTH-002",
            payer="Synthetic Health Plan",
            amount=920.00,
            carc="96",
            rarc="N130",
            payer_explanation="This charge is not covered under the member's benefit plan.",
            documentation_summary=STRONG_DOCUMENTATION,
        ),
        "escalate",
    ),
    (
        "unmapped denial code",
        DenialRecord(
            claim_id="CLM-100044",
            patient_id="SYNTH-003",
            payer="Synthetic Health Plan",
            amount=310.00,
            carc="99",
            rarc=None,
            payer_explanation="Denied. See remittance advice for details.",
            documentation_summary=STRONG_DOCUMENTATION,
        ),
        "escalate",
    ),
    (
        "non-covered charge with contradicting evidence",
        DenialRecord(
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
        ),
        "escalate",
    ),
    (
        "authorization missing, PA referenced in notes",
        DenialRecord(
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
        ),
        None,  # open case - whatever it does, the trace is the point
    ),
]


def main() -> None:
    agent = DenialAppealAgent(
        code_lookup=DenialCodeLookup(),
        model=ModelClient(),
    )

    mismatches = 0
    checked = 0

    for label, denial, expected in CASES:
        print("=" * 72)
        header = f"CASE: {label}"
        if expected:
            header += f"   (expecting: {expected})"
        print(header)
        print("=" * 72)

        state = agent.run(denial)
        actual = state.decision.value if state.decision else "none"

        for line in state.trace:
            print(line)

        print(f"\ndecision:    {actual}")
        print(f"stop_reason: {state.stop_reason}")
        print(f"tools used:  {', '.join(state.tools_called) or '(none)'}")

        if expected:
            checked += 1
            if actual != expected:
                mismatches += 1
                print(f"\n*** MISMATCH: expected {expected}, got {actual}")

        print()

    print("=" * 72)
    if mismatches:
        print(f"{mismatches} of {checked} checked cases did not behave as expected.")
        print("Read the traces above before changing anything.")
    else:
        print(f"All {checked} checked cases behaved as expected. Day 2 complete.")


if __name__ == "__main__":
    main()