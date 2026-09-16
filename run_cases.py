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

# Day 15. The claim definitions moved to cases.py. They lived here and in
# calibrate_tools.py with different fields, so the same claim ID meant two
# different things depending on which file a test imported from.
import sys

from cases import ALL as _ALL, LABELS as _LABELS

_EXPECTED = {
    "CLM-100042": "appeal",
    "CLM-100043": "do_not_appeal",
    "CLM-100044": "escalate",
    "CLM-100045": "escalate",
    "CLM-100046": None,  # open case - whatever it does, the trace is the point
}

CASES = [(_LABELS[c.claim_id], c, _EXPECTED[c.claim_id]) for c in _ALL]


# Day 18. Which provider to run against. Without this the script always used
# the default, which is the one whose daily quota runs out first.
#
#     python run_cases.py            the default provider
#     python run_cases.py groq       an explicit one
_PROVIDER = sys.argv[1] if len(sys.argv) > 1 else None


def main() -> None:
    agent = DenialAppealAgent(
        code_lookup=DenialCodeLookup(),
        model=ModelClient(provider=_PROVIDER),
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
