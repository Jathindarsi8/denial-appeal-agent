"""
Day 1 verification: three cases that should take three different paths.

    1. CARC 50, strong documentation  -> appeal authorized
    2. CARC 96, strong documentation  -> model may propose appeal, guardrail refuses
    3. CARC 99, unmapped              -> escalates before the model is ever called

Case 2 is the important one. It is where the deterministic layer overrides a
plausible-looking model judgment.
"""

from agent import (
    DenialAppealAgent,
    DenialCodeLookup,
    DenialRecord,
    JudgeModel,
    PolicyRetriever,
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
            payer_explanation=(
                "This service is not covered under the member's benefit plan."
            ),
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
]


def main() -> None:
    agent = DenialAppealAgent(
        code_lookup=DenialCodeLookup(),
        judge=JudgeModel(),
        policy_retriever=PolicyRetriever(),
    )

    failures = 0

    for label, denial, expected in CASES:
        print("=" * 70)
        print(f"CASE: {label}   (expecting: {expected})")
        print("=" * 70)

        state = agent.run(denial)
        actual = state.decision.value if state.decision else "none"

        for line in state.trace:
            print(line)

        print(f"\ndecision:    {actual}")
        print(f"stop_reason: {state.stop_reason}")

        if actual != expected:
            failures += 1
            print(f"\n*** MISMATCH: expected {expected}, got {actual}")

        print()

    print("=" * 70)
    if failures:
        print(f"{failures} of {len(CASES)} cases did not behave as expected.")
        print("Read the trace above before changing anything -- the disagreement")
        print("is information, not necessarily a bug in the guardrails.")
    else:
        print(f"All {len(CASES)} cases behaved as expected. Day 1 complete.")


if __name__ == "__main__":
    main()
