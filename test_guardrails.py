"""
The guardrail suite. Runs offline, in about a second, before every commit.

Started on day 10 to reach six rules that had never executed in nineteen real
runs. Grown since: day 11 added the web-provenance rules, day 13 added the
required-checks rules. The point has not changed — a rule that has never run is
a rule that might not work, and these are the rules that exist to stop
something.

Day 10, the original problem:

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
          claim_id="TEST-001", patient_id="TEST-PT",
          procedure_code=None, date_of_service=None) -> DenialRecord:
    return DenialRecord(
        claim_id=claim_id,
        patient_id=patient_id,
        payer="Test Plan",
        amount=100.0,
        carc=carc,
        rarc=None,
        payer_explanation="Test denial.",
        documentation_summary=docs,
        procedure_code=procedure_code,
        date_of_service=date_of_service,
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

    print("Guardrail suite. 26 checks, no API calls.\n")

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
    # Both required checks are called, so the run reaches the confidence rule
    # rather than stopping earlier on required_checks_not_run.
    low_conf = ScriptedModel([call_tool("retrieve_policy"),
                              call_tool("check_prior_authorization"),
                              judge(confidence=CONFIDENCE_FLOOR - 0.1)])
    check("confidence under the floor escalates",
          run(low_conf, claim()),
          Decision.ESCALATE, "confidence_below_floor", verbose)

    # And the boundary. A rule tested only in the middle of its range is a rule
    # whose edge is untested, and thresholds fail at the edge.
    at_floor = ScriptedModel([call_tool("retrieve_policy"),
                              call_tool("check_prior_authorization"),
                              judge(confidence=CONFIDENCE_FLOOR)])
    check("confidence exactly at the floor is allowed through",
          run(at_floor, claim()),
          Decision.APPEAL, "appeal_authorized", verbose)

    # ── 6. contradicting a human decision
    # Needs a recorded human resolution, so it writes one, tests, and removes
    # it. This rule was added on day 7 and has never run, because nobody has
    # ever recorded a resolution.
    import store, io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        store.init()  # setup, not a result
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

    # Day 11 changed this deliberately. An unmapped code used to stop before
    # the model. It now lets the model run so it can look the code up and hand
    # a human some context, but the outcome is still capped at escalate. Both
    # halves are asserted: the outcome and the cap.
    unmapped = ScriptedModel(judge())
    check("an unmapped code escalates on unverified provenance",
          run(unmapped, claim(carc="99999")),
          Decision.ESCALATE, "unverified_code_definition", verbose)

    # With web lookup off, the old behaviour must still hold, because that is
    # what runs when the search dependency is missing or disabled.
    unmapped_no_web = ScriptedModel(judge())
    agent = DenialAppealAgent(
        code_lookup=DenialCodeLookup(), model=unmapped_no_web,
        audit_log=False, use_memory=False, resume=False, web_lookup=False)
    check("with web lookup off, an unmapped code stops before the model",
          agent.run(claim(carc="99999")),
          Decision.ESCALATE, "unmapped_denial_code", verbose)
    if unmapped_no_web.calls != 0:
        print("        NOTE: the model was called with web lookup disabled. "
              "The lookup is supposed to stop before that.")

    # The web tool must not be reachable on a code the curated table knows.
    # Otherwise a public search could be used to argue against a definition
    # that was deliberately curated.
    tries_search = ScriptedModel([call_tool("search_denial_code"),
                                  call_tool("retrieve_policy"), judge()])
    state = run(tries_search, claim(carc="197"))
    refused = any("refused" in line and "search_denial_code" in line
                  for line in state.trace)
    RESULTS.append(("web search is refused on a mapped code", refused,
                    "refused" if refused else "ALLOWED"))
    print(f"  {'PASS' if refused else 'FAIL'}  "
          f"web search is refused on a mapped code")
    if not refused:
        for line in state.trace:
            print(f"        {line}")

    # An appeal can never come out of a web-derived category, whatever the
    # model proposes or how confident it is.
    web_appeal = ScriptedModel([call_tool("search_denial_code"),
                                judge(confidence=0.99)])
    check("a web-derived category cannot authorise an appeal",
          run(web_appeal, claim(carc="99999")),
          Decision.ESCALATE, "unverified_code_definition", verbose)

    never_appeal = ScriptedModel([call_tool("retrieve_policy"),
                                  judge(category="noncovered_charge")])
    check("a non-covered charge still cannot be auto-appealed",
          run(never_appeal, claim(carc="96")),
          Decision.ESCALATE, "category_requires_human_review", verbose)

    # Moving the checks into the loop removed the scenario. A model can no longer reach a judgment with
    # nothing retrieved on a mapped code, because the required checks run
    # before its first turn. What is asserted now is that outcome.
    no_evidence = ScriptedModel(judge())
    check("a model that asks for nothing still decides on gathered evidence",
          run(no_evidence, claim()),
          Decision.APPEAL, "appeal_authorized", verbose)

    # ── 8. Day 13: required checks
    #
    # Day 13 required these before either terminal decision, after a second
    # provider closed a claim at 0.96 without checking whether the
    # authorization it hinged on existed. Later the same day they moved into the loop, so
    # the model no longer gets the chance to skip them and the scenarios the
    # original tests scripted can no longer occur.
    #
    # The rule still exists, so it still needs a test that reaches it. The
    # backstop covers one real gap: a category listed in REQUIRED_TOOLS whose
    # tool is missing from TOOLS. Nothing would run it, and without the rule
    # the claim would be decided on a check that silently never happened.
    print()

    import agent as agent_module
    original = dict(agent_module.REQUIRED_TOOLS)
    agent_module.REQUIRED_TOOLS["authorization_missing"] = {
        "retrieve_policy", "a_tool_that_does_not_exist"}
    try:
        gap = ScriptedModel(judge(decision="do_not_appeal", confidence=0.96))
        check("a required check with no implementation still blocks a decision",
              run(gap, claim(carc="197")),
              Decision.ESCALATE, "required_checks_not_run", verbose)
    finally:
        agent_module.REQUIRED_TOOLS.clear()
        agent_module.REQUIRED_TOOLS.update(original)

    # Escalating is a handoff, not a conclusion, so it needs nothing.
    escalates_bare = ScriptedModel(judge(decision="escalate"))
    check("escalating is allowed regardless of what ran",
          run(escalates_bare, claim(carc="197")),
          Decision.ESCALATE, "model_requested_escalation", verbose)

    # The checks run before the model's first turn, so a model that
    # asks for nothing at all still arrives at a decision with the evidence
    # already gathered.
    asks_for_nothing = ScriptedModel(judge(decision="do_not_appeal"))
    state = run(asks_for_nothing, claim(carc="197"))
    check("required checks run without the model asking",
          state, Decision.DO_NOT_APPEAL, "denial_appears_correct_on_record",
          verbose)

    ran_automatically = any("run automatically" in line for line in state.trace)
    RESULTS.append(("the trace records checks as automatic, not chosen",
                    ran_automatically,
                    "recorded" if ran_automatically else "NOT RECORDED"))
    print(f"  {'PASS' if ran_automatically else 'FAIL'}  "
          f"the trace records checks as automatic, not chosen")

    # A model that asks for a required tool anyway gets refused as a repeat,
    # because the loop already ran it.
    asks_anyway = ScriptedModel([call_tool("check_prior_authorization"),
                                 judge(decision="do_not_appeal")])
    state = run(asks_anyway, claim(carc="197"))
    refused = any("repeated tool" in line for line in state.trace)
    RESULTS.append(("asking for an already-run check is refused as a repeat",
                    refused, "refused" if refused else "ALLOWED"))
    print(f"  {'PASS' if refused else 'FAIL'}  "
          f"asking for an already-run check is refused as a repeat")

    # ── 9. Day 13: the authorization check verifies rather than echoes
    # This tool was a substring search on the claim notes for twelve days. It
    # reported back what the model had already read and could not fail, which
    # made calling it a ceremony and made day 13's rule requiring it wrong.
    # These assert it can now contradict the claim text.
    print()
    import authorizations, io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        authorizations.seed()  # setup, not a result

    from agent import check_prior_authorization

    # The notes assert an authorization that was never issued. The old stub
    # said "an authorization is referenced" because the string was present.
    ghost = check_prior_authorization(
        claim(docs="Prior authorization PA-99999 was obtained before service.",
              patient_id="SYNTH-009"), "authorization_missing")
    ok = "DOES NOT EXIST" in ghost
    RESULTS.append(("an authorization only the notes believe in is refuted",
                    ok, ghost[:60]))
    print(f"  {'PASS' if ok else 'FAIL'}  "
          f"an authorization only the notes believe in is refuted")
    if not ok:
        print(f"        got: {ghost[:200]}")

    # Exists and approved, but authorises a different procedure. The notes are
    # telling the truth and the authorization still does not help.
    mismatch = check_prior_authorization(
        claim(docs="Prior authorization PA-88213 was approved.",
              patient_id="SYNTH-004", procedure_code="29827",
              date_of_service="2026-06-20"), "authorization_missing")
    ok = "PROBLEMS FOUND" in mismatch and "64483" in mismatch
    RESULTS.append(("a procedure mismatch is caught", ok, mismatch[:60]))
    print(f"  {'PASS' if ok else 'FAIL'}  a procedure mismatch is caught")
    if not ok:
        print(f"        got: {mismatch[:200]}")

    # Everything lines up.
    clean = check_prior_authorization(
        claim(docs="Scheduling notes reference PA-77104.",
              patient_id="SYNTH-005", procedure_code="29827",
              date_of_service="2026-06-20"), "authorization_missing")
    ok = "No discrepancies" in clean
    RESULTS.append(("a matching authorization is confirmed", ok, clean[:60]))
    print(f"  {'PASS' if ok else 'FAIL'}  a matching authorization is confirmed")
    if not ok:
        print(f"        got: {clean[:200]}")

    # ── 10. Day 17: closing a claim against verified evidence
    # Every one of the seventeen failures in the day 16 evaluation was the
    # model closing a claim and nothing stopping it. These assert the rule
    # that now stops it, and equally that it does not fire where closing is
    # the right answer.
    print()
    from cases import CLM_100045, CLM_100046, CLM_100043

    closes_verified = ScriptedModel(judge(decision="do_not_appeal",
                                          category="authorization_missing"))
    check("closing a claim with a verified authorization escalates",
          run(closes_verified, CLM_100046),
          Decision.ESCALATE, "closed_against_verified_evidence", verbose)

    closes_conflict = ScriptedModel(judge(decision="do_not_appeal",
                                          category="noncovered_charge"))
    check("closing a non-covered claim with an approved auth escalates",
          run(closes_conflict, CLM_100045),
          Decision.ESCALATE, "closed_against_verified_evidence", verbose)

    # The rule must not fire where closing is correct, or it converts a
    # measured failure into a different measured failure.
    closes_correctly = ScriptedModel(judge(decision="do_not_appeal",
                                           category="noncovered_charge"))
    check("closing a claim with no authorization on file is still allowed",
          run(closes_correctly, CLM_100043),
          Decision.DO_NOT_APPEAL, "denial_appears_correct_on_record", verbose)

    # An authorization the notes invent but the system never issued supports
    # the denial. Closing stays available.
    ghost_claim = claim(carc="197", patient_id="SYNTH-009",
                        docs="Prior authorization PA-99999 was obtained.")
    closes_on_ghost = ScriptedModel(judge(decision="do_not_appeal"))
    check("closing on an authorization that does not exist is allowed",
          run(closes_on_ghost, ghost_claim),
          Decision.DO_NOT_APPEAL, "denial_appears_correct_on_record", verbose)

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
    print("\nEvery rule has at least one check behind it.")


if __name__ == "__main__":
    main()
