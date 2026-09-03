"""
Day 10: test the guardrails that have never fired.

Nineteen runs, six rules decided them. There are twelve rules. The other six
have never executed once:

    model_output_failed_validation
    step_limit_reached
    category_disagreement
    contradicts_human_decision
    appeal_proposed_without_supporting_documentation
    confidence_below_floor

Six rules that have never run are six rules that might not work. They are also
the ones that matter most, because each exists to stop something bad, and the
moment you find out a stop does not stop is the moment it was needed.

They cannot be reached with real calls. You would have to wait for a model to
misbehave in a specific way, on a claim shaped to expose it, on a free tier that
allows twenty attempts a day. So this drives the agent with a scripted model
that returns exactly what each test needs.

No API calls. Runs in under a second. Should run before every commit.

    python test_guardrails.py
    python test_guardrails.py -v      show the trace for each case
"""

from __future__ import annotations

import sys

from agent import (
    CONFIDENCE_FLOOR,
    Decision,
    DenialAppealAgent,
    DenialCodeLookup,
    DenialRecord,
    ModelAction,
)


class ScriptedModel:
    """Stands in for ModelClient. Returns a fixed list of actions, one per
    turn. Same interface as the real thing, so the agent cannot tell."""

    def __init__(self, actions: list[ModelAction] | ModelAction,
                 model: str = "scripted", raises: Exception | None = None):
        self.actions = actions if isinstance(actions, list) else [actions]
        self.model = model
        self.raises = raises
        self.calls = 0

    def step(self, messages: list[dict]) -> ModelAction:
        self.calls += 1
        if self.raises:
            raise self.raises
        # Past the end of the script, repeat the last action. That is what
        # produces a loop, which is how the step limit gets tested.
        idx = min(self.calls - 1, len(self.actions) - 1)
        return self.actions[idx]


def claim(carc="197", docs="PA-77104 referenced in scheduling notes.",
          claim_id="TEST-001") -> DenialRecord:
    return DenialRecord(
        claim_id=claim_id,
        patient_id="TEST-PT",
        payer="Test Plan",
        amount=100.0,
        carc=carc,
        rarc=None,
        payer_explanation="Test denial.",
        documentation_summary=docs,
    )


def judge(decision="appeal", category="authorization_missing",
          confidence=0.9) -> ModelAction:
    return ModelAction(
        action="judge",
        denial_category=category,
        root_cause="test",
        proposed_decision=decision,
        confidence=confidence,
        reasoning_summary="test",
    )


def call_tool(tool="retrieve_policy") -> ModelAction:
    return ModelAction(action="call_tool", tool=tool, tool_reason="test")


def run(model, denial, **kw):
    """Every test runs with audit, memory and resume off. A test must not write
    to the run log, read history that changes between runs, or resume a
    checkpoint from a previous test."""
    agent = DenialAppealAgent(
        code_lookup=DenialCodeLookup(),
        model=model,
        audit_log=False,
        use_memory=False,
        resume=False,
        **kw,
    )
    return agent.run(denial)


# ─────────────────────────────────────────────────────────── the tests

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, state, expect_decision: Decision, expect_reason: str,
          verbose: bool = False) -> None:
    ok = (state.decision == expect_decision
          and state.stop_reason.startswith(expect_reason))
    detail = f"{state.decision.value if state.decision else None} / {state.stop_reason}"
    RESULTS.append((name, ok, detail))

    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}")
    if not ok:
        print(f"        expected {expect_decision.value} / {expect_reason}")
        print(f"        got      {detail}")
    if verbose:
        for line in state.trace:
            print(f"        {line}")
        print()


def main() -> None:
    verbose = "-v" in sys.argv

    print("Guardrails that have never fired in a real run.\n")

    # ── 1. malformed model output
    # The model returns something that is not a valid ModelAction. Pydantic
    # raises, and the agent must escalate rather than crash. This is the path
    # that runs when a provider returns truncated or non-JSON output, which
    # happens and has not happened here yet.
    from pydantic import ValidationError
    try:
        ModelAction.model_validate({"action": "not_a_real_action"})
    except ValidationError as exc:
        bad_model = ScriptedModel([], raises=exc)
        check("malformed model output escalates",
              run(bad_model, claim()),
              Decision.ESCALATE, "model_output_failed_validation", verbose)

    # ── 2. step limit
    # A model that only ever asks for tools never reaches a judgment. Without a
    # bound this is an infinite loop that spends the whole daily budget.
    looping = ScriptedModel([call_tool("retrieve_policy"),
                             call_tool("check_prior_authorization"),
                             call_tool("retrieve_policy")])
    check("endless tool calls hit the step limit",
          run(looping, claim(), max_steps=4),
          Decision.ESCALATE, "step_limit_reached", verbose)

    # ── 3. category disagreement
    # The code table says authorization_missing. The model says medical
    # necessity. One of them is wrong and the agent cannot tell which, so a
    # human has to. This is the rule that stops an appeal being argued on the
    # wrong basis, which policy AP-01 says is dismissed without review.
    wrong_category = ScriptedModel(judge(category="medical_necessity"))
    check("model disagreeing with the code table escalates",
          run(wrong_category, claim(carc="197")),
          Decision.ESCALATE, "category_disagreement", verbose)

    # ── 4. no documentation on file
    # An appeal with nothing behind it. Should never be authorised regardless
    # of what the model claims.
    no_docs = ScriptedModel([call_tool("retrieve_policy"), judge()])
    check("appeal with no documentation escalates",
          run(no_docs, claim(docs="   ")),
          Decision.ESCALATE, "appeal_proposed_without_supporting_documentation",
          verbose)

    # ── 5. confidence below the floor
    # Zero of nineteen real runs have exercised this. The floor is the most
    # discussed rule in this project and the least tested.
    low_conf = ScriptedModel([call_tool("retrieve_policy"),
                              judge(confidence=CONFIDENCE_FLOOR - 0.1)])
    check("confidence under the floor escalates",
          run(low_conf, claim()),
          Decision.ESCALATE, "confidence_below_floor", verbose)

    # And the boundary. A rule tested only in the middle of its range is a rule
    # whose edge is untested, and thresholds fail at the edge.
    at_floor = ScriptedModel([call_tool("retrieve_policy"),
                              judge(confidence=CONFIDENCE_FLOOR)])
    check("confidence exactly at the floor is allowed through",
          run(at_floor, claim()),
          Decision.APPEAL, "appeal_authorized", verbose)

    # ── 6. contradicting a human decision
    # Needs a recorded human resolution, so it writes one, tests, and removes
    # it. This rule was added on day 7 and has never run, because nobody has
    # ever recorded a resolution.
    import store
    store.init()
    store.record_resolution("TEST-HUMAN", "reviewer@test",
                            "do_not_appeal", "test fixture")
    try:
        contradicts = ScriptedModel([call_tool("retrieve_policy"), judge()])
        agent = DenialAppealAgent(
            code_lookup=DenialCodeLookup(), model=contradicts,
            audit_log=False, use_memory=True, resume=False)
        state = agent.run(claim(claim_id="TEST-HUMAN"))
        check("model contradicting a human decision escalates",
              state, Decision.ESCALATE, "contradicts_human_decision", verbose)
    finally:
        with store.connect() as conn:
            conn.execute("DELETE FROM resolutions WHERE claim_id = 'TEST-HUMAN'")
            conn.commit()

    # ── 7. rules that DO fire, as a regression check
    # If a change to the code above quietly breaks the ordinary path, these
    # catch it.
    print()
    normal = ScriptedModel([call_tool("retrieve_policy"),
                            call_tool("check_prior_authorization"), judge()])
    check("a well-supported appeal is still authorised",
          run(normal, claim()),
          Decision.APPEAL, "appeal_authorized", verbose)

    unmapped = ScriptedModel(judge())
    check("an unmapped denial code still escalates before the model",
          run(unmapped, claim(carc="99999")),
          Decision.ESCALATE, "unmapped_denial_code", verbose)
    if unmapped.calls != 0:
        print("        NOTE: the model was called on an unmapped code. "
              "The lookup is supposed to stop before that.")

    never_appeal = ScriptedModel([call_tool("retrieve_policy"),
                                  judge(category="noncovered_charge")])
    check("a non-covered charge still cannot be auto-appealed",
          run(never_appeal, claim(carc="96")),
          Decision.ESCALATE, "category_requires_human_review", verbose)

    no_evidence = ScriptedModel(judge())
    check("an appeal with nothing retrieved still escalates",
          run(no_evidence, claim()),
          Decision.ESCALATE, "appeal_proposed_without_retrieved_evidence",
          verbose)

    # ── summary
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 68)
    print(f"{passed} of {len(RESULTS)} passed")
    if passed < len(RESULTS):
        print("\nfailures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  {name}")
                print(f"    got {detail}")
        sys.exit(1)
    print("\nEvery guardrail now has at least one run behind it.")


if __name__ == "__main__":
    main()
