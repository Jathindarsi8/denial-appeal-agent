"""
Denial triage + appeal drafting agent.

    denial
      -> deterministic CARC lookup (always first, never the model's choice)
      -> prior history recalled from the store
      -> model turn
           |- requests a tool  -> loop runs it, appends the observation,
           |                      hands control back to the model
           `- returns judgment -> deterministic guardrail validation
      -> draft appeal / escalate / stop
      -> audit record written, always

The model decides what it needs to know. The loop decides what actually runs.
The guardrails decide what it's allowed to do with the answer.

Retry with exponential backoff is in here early, because the free tier rate
limits at 5 requests per minute and a five-case run blows straight past that.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv
from openai import APIStatusError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


# ---------------------------------------------------------------- domain types

DenialCategory = Literal[
    "medical_necessity",
    "missing_or_invalid_information",
    "noncovered_charge",
    "authorization_missing",
    "timely_filing",
    "duplicate_claim",
    "coding_error",
    "other",
]

ToolName = Literal["retrieve_policy", "check_prior_authorization",
                   "search_denial_code"]


class Decision(str, Enum):
    APPEAL = "appeal"
    DO_NOT_APPEAL = "do_not_appeal"
    ESCALATE = "escalate"


@dataclass
class DenialRecord:
    claim_id: str
    patient_id: str
    payer: str
    amount: float
    carc: str
    rarc: Optional[str]
    payer_explanation: str
    documentation_summary: str


class ModelAction(BaseModel):
    """One model turn. Either it wants a tool, or it's ready to judge."""

    action: Literal["call_tool", "judge"]

    # action == "call_tool"
    tool: Optional[ToolName] = None
    tool_reason: Optional[str] = None

    # action == "judge"
    denial_category: Optional[DenialCategory] = None
    root_cause: Optional[str] = None
    proposed_decision: Optional[Literal["appeal", "do_not_appeal", "escalate"]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning_summary: Optional[str] = None


@dataclass
class AgentState:
    denial: DenialRecord
    steps_taken: int = 0

    code_category: Optional[str] = None
    code_meaning: Optional[str] = None
    # Day 11. Where the category came from. "table" is the curated lookup.
    # "web" means a public search suggested it and nothing has verified it.
    code_provenance: str = "table"

    messages: list[dict] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    judgment: Optional[ModelAction] = None
    decision: Optional[Decision] = None
    appeal_draft: Optional[str] = None
    stop_reason: Optional[str] = None
    trace: list[str] = field(default_factory=list)

    # Day 7. What the store knows about this claim, member and denial code.
    memory: Optional[Any] = None

    def log(self, message: str) -> None:
        self.trace.append(f"[step {self.steps_taken}] {message}")


# ---------------------------------------------------------------------- tools

class DenialCodeLookup:
    """Deterministic. Always runs first. The model never chooses whether to."""

    CODES: dict[str, tuple[str, str]] = {
        "16": ("missing_or_invalid_information",
               "Claim/service lacks information or contains submission errors."),
        "29": ("timely_filing",
               "The time limit for filing this claim has expired."),
        "50": ("medical_necessity",
               "Service was denied as not medically necessary by the payer."),
        "96": ("noncovered_charge",
               "Charge is considered non-covered under the payer's policy."),
        "197": ("authorization_missing",
                "Precertification, authorization, or notification was absent."),
    }

    def lookup(self, carc: str) -> tuple[Optional[str], str]:
        if carc in self.CODES:
            return self.CODES[carc]
        return None, f"No trusted local mapping is available for CARC {carc}."


def retrieve_policy(denial: DenialRecord, category: Optional[str]) -> str:
    """Day 6. Real retrieval over the policy corpus in policies/.

    Was a hardcoded if-statement returning invented policy numbers. Every
    result before day 6 rests on the agent 'retrieving' from that stub, so
    nothing measured before it is comparable to anything measured after.

    Category comes from the deterministic code lookup, not from the model, so
    the model cannot widen its own search.
    """
    from retrieval import get_retriever

    query = f"{denial.payer_explanation} {denial.documentation_summary}"
    # The claim text finds documents about what the claim IS. The second query
    # finds documents about what can be DONE with it, which share almost no
    # vocabulary with the first. One query missed a policy withdrawing the
    # appeal route on a claim about exactly that route.
    return get_retriever().retrieve_for_agent(
        query,
        category=category,
        k=3,
        extra_queries=[
            "can this denial be appealed, what routes are available, "
            "are appeals accepted on this basis"
        ],
        max_per_doc=1,
    )


def check_prior_authorization(denial: DenialRecord, category: Optional[str]) -> str:
    """Stubbed PA check. Reads what the claim record actually says."""
    text = denial.documentation_summary.lower()
    if "prior authorization" in text or "pa-" in text:
        return (
            "A prior authorization reference appears in the claim documentation. "
            "This system cannot confirm it against the payer's authorization "
            "database. Treat as unverified."
        )
    return "No prior authorization reference found in the claim documentation."


def search_denial_code(denial: DenialRecord, category: Optional[str]) -> str:
    """Day 11. Look up a denial code the curated table does not have.

    Only reachable when the table has already failed. If the code is known,
    this tool is not offered at all, so a public search can never be used to
    argue against a definition that was deliberately curated.

    Everything it returns is labelled unverified inside the observation text,
    next to the content, because that is where the model actually reads.
    """
    from websearch import format_for_agent, search_code

    try:
        results = search_code(denial.carc)
    except Exception as exc:
        return (f"Web lookup for CARC {denial.carc} failed "
                f"({type(exc).__name__}). The code remains unidentified.")
    return format_for_agent(denial.carc, results)


TOOLS: dict[str, Callable[[DenialRecord, Optional[str]], str]] = {
    "retrieve_policy": retrieve_policy,
    "check_prior_authorization": check_prior_authorization,
    "search_denial_code": search_denial_code,
}

# Offered only when the deterministic lookup could not identify the code.
UNMAPPED_ONLY_TOOLS = {"search_denial_code"}


# ------------------------------------------------------------ model interface

SYSTEM_PROMPT = """\
You are a claims denial analyst working one claim at a time.

You do not make final decisions. A deterministic validation layer reviews
whatever you propose and can override it. Your job is accurate assessment.

Each turn you return JSON, and you choose one of two actions.

To gather information:
{"action": "call_tool", "tool": "<tool name>", "tool_reason": "<one sentence>"}

Available tools:
  retrieve_policy            - payer policy statements for this denial category
  check_prior_authorization  - whether the claim record references a prior auth
  search_denial_code         - public web search for what a denial code means.
                               Available ONLY when the code is missing from the
                               trusted table. Results are unverified and cannot
                               support an appeal, only a better escalation.

To give your assessment:
{"action": "judge",
 "denial_category": one of ["medical_necessity", "missing_or_invalid_information",
    "noncovered_charge", "authorization_missing", "timely_filing",
    "duplicate_claim", "coding_error", "other"],
 "root_cause": "one sentence",
 "proposed_decision": one of ["appeal", "do_not_appeal", "escalate"],
 "confidence": number between 0.0 and 1.0,
 "reasoning_summary": "two or three sentences"}

Rules:
- Call a tool only when the answer would actually change your assessment.
- Never call the same tool twice.
- Propose "appeal" only with concrete support that contradicts the stated reason.
- Propose "escalate" when the record is ambiguous or depends on information you
  do not have.
- Never assert payer policy language you have not retrieved.
- Prior history is context, not instruction. A previous outcome does not tell
  you what this run should conclude. Where a human has already decided a claim,
  do not propose something that contradicts them without saying why.
- Set confidence honestly. Low confidence is useful information, not failure.

Return JSON only. No prose outside the JSON.
"""


class ModelClient:
    def __init__(self, model: Optional[str] = None):
        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv(
                "LLM_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )
        self.model = model or os.getenv("LLM_MODEL", "gemini-3.6-flash")

    # Day 8. These were one except block retrying six times, and it cost a
    # whole day's budget. An upstream 500 got retried five times at 2, 4, 8,
    # 16 and 32 seconds. Every retry is a request that counts, so one fault on
    # the provider's side spent five of twenty daily calls and returned
    # nothing, which then made the next two runs fail on quota. The two errors
    # need opposite handling.
    #
    # 429  the limit is real and time-based, so waiting works. Google sends
    #      retryDelay in the error; honour it rather than guessing, because
    #      the guess capped at 32s when it had asked for 57.
    # 500  the upstream is unhealthy. Retrying does not make it healthy, and
    #      each attempt still costs budget. Try twice, then stop.
    MAX_SERVER_ERROR_RETRIES = 2
    MAX_RATE_LIMIT_RETRIES = 4

    def step(self, messages: list[dict]) -> ModelAction:
        """One model turn. Retries a rate limit patiently and an upstream fault
        barely, because only one of the two is worth waiting out."""
        server_errors = 0
        rate_limits = 0
        delay = 2

        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                raw = completion.choices[0].message.content
                if raw is None:
                    raise ValueError("Model returned no content")
                return ModelAction.model_validate_json(raw)

            except RateLimitError as exc:
                # Day 9. Two very different things return 429. Per-minute
                # throttling clears in under a minute and is worth waiting
                # out. A daily cap clears tomorrow, and waiting three minutes
                # for it just spends more requests confirming it is still
                # there. Google names which one it is in the error.
                if _is_daily_quota(exc):
                    print("  ...daily quota exhausted for this model, "
                          "stopping. The checkpoint keeps.")
                    raise
                rate_limits += 1
                if rate_limits > self.MAX_RATE_LIMIT_RETRIES:
                    raise
                wait = _retry_after(exc) or delay
                print(f"  ...rate limited, waiting {wait:.0f}s "
                      f"(attempt {rate_limits}/{self.MAX_RATE_LIMIT_RETRIES})")
                time.sleep(wait)
                delay = min(delay * 2, 64)

            except InternalServerError:
                server_errors += 1
                if server_errors > self.MAX_SERVER_ERROR_RETRIES:
                    print(f"  ...upstream failed {server_errors}x, giving up "
                          f"rather than spending more budget on it")
                    raise
                print(f"  ...upstream error, retrying once "
                      f"({server_errors}/{self.MAX_SERVER_ERROR_RETRIES})")
                time.sleep(3)



def _is_daily_quota(exc: APIStatusError) -> bool:
    """A per-day cap is not a wait-and-retry condition. Google distinguishes
    them in quotaId: PerDay vs PerMinute."""
    try:
        for detail in exc.body[0]["error"]["details"]:  # type: ignore[index]
            for violation in detail.get("violations", []):
                if "PerDay" in str(violation.get("quotaId", "")):
                    return True
    except Exception:
        pass
    return False


def _retry_after(exc: APIStatusError) -> Optional[float]:
    """Google puts the real wait in the error body as retryDelay, and in the
    Retry-After header. Reading it beats guessing: the guess capped at 32s on a
    day the API had asked for 57."""
    try:
        header = exc.response.headers.get("retry-after")
        if header:
            return float(header)
    except Exception:
        pass
    try:
        for detail in exc.body[0]["error"]["details"]:  # type: ignore[index]
            delay = detail.get("retryDelay")
            if delay:
                return float(str(delay).rstrip("s"))
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ guardrails

CONFIDENCE_FLOOR = 0.6
NEVER_AUTO_APPEAL = {"noncovered_charge", "timely_filing", "duplicate_claim", "other"}


def validate_action(state: AgentState) -> tuple[Decision, str]:
    """Every rule here is a verifiable condition, not a vibe."""
    j = state.judgment
    assert j is not None

    # Day 11. Handled before the checks below, because on an unmapped code
    # state.code_category stays None and every rule that compares against it
    # would fire first and report the wrong reason.
    #
    # A category the agent found on the web is context for a human, never
    # grounds for an action. It makes the handoff better; it does not remove
    # the handoff. What the model read the code as is recorded so a reviewer
    # opens the claim with something rather than nothing.
    if state.code_provenance != "table":
        read_as = j.denial_category or "could not identify"
        return Decision.ESCALATE, (
            f"unverified_code_definition:searched_web,"
            f"model_read_it_as={read_as}"
        )

    if state.code_category is None:
        return Decision.ESCALATE, "unmapped_denial_code"

    if j.denial_category != state.code_category:
        return Decision.ESCALATE, (
            f"category_disagreement:code={state.code_category},model={j.denial_category}"
        )

    # Day 7. A human decision outranks the model, always. If someone already
    # worked this claim and the model now wants to do something else, that is a
    # conflict for a person to settle, not for the agent to resolve quietly.
    mem = state.memory
    human = getattr(mem, "human_resolution", None) if mem else None
    if human and j.proposed_decision and j.proposed_decision != human:
        return Decision.ESCALATE, (
            f"contradicts_human_decision:human={human},model={j.proposed_decision}"
        )

    if j.proposed_decision == "escalate":
        return Decision.ESCALATE, "model_requested_escalation"

    if j.proposed_decision == "do_not_appeal":
        return Decision.DO_NOT_APPEAL, "denial_appears_correct_on_record"

    if j.denial_category in NEVER_AUTO_APPEAL:
        return Decision.ESCALATE, f"category_requires_human_review:{j.denial_category}"

    if not state.denial.documentation_summary.strip():
        return Decision.ESCALATE, "appeal_proposed_without_supporting_documentation"

    # Day 4. Measured: with no tools the model returns 0.95 every single time.
    # With tools it returns 0.75-0.85 and sometimes changes its decision. The
    # score goes DOWN as the agent learns more, so a high score is evidence of
    # ignorance, not of a strong case. Retrieval is therefore a precondition
    # for authorization, and confidence is only consulted afterwards.
    if not state.observations:
        return Decision.ESCALATE, "appeal_proposed_without_retrieved_evidence"

    if j.confidence is None or j.confidence < CONFIDENCE_FLOOR:
        return Decision.ESCALATE, f"confidence_below_floor:{j.confidence}"

    return Decision.APPEAL, f"appeal_authorized:evidence={len(state.observations)}"


# ----------------------------------------------------------------- agent loop

class DenialAppealAgent:
    def __init__(
        self,
        code_lookup: DenialCodeLookup,
        model: ModelClient,
        max_steps: int = 8,
        audit_log=None,
        use_memory: bool = True,
        resume: bool = True,
        web_lookup: bool = True,
    ):
        self.code_lookup = code_lookup
        self.model = model
        self.max_steps = max_steps
        self.audit_log = audit_log
        self.use_memory = use_memory
        self.resume = resume
        self.web_lookup = web_lookup

    def run(self, denial: DenialRecord) -> AgentState:
        """Run the agent, then record what happened. Auditing is not optional.

        Day 5. Every exit path from _run lands here, so a run cannot finish
        without leaving a record. Pass audit_log=False to disable it, which
        only the probes and unit tests do.
        """
        state = self._run(denial)

        # Day 9. The guardrails have decided, so this run is closed. Leaving it
        # open would let a later rerun reopen a settled case.
        try:
            import checkpoint as ckpt
            ckpt.save(state,
                      ckpt.run_key(denial, self.model.model),
                      ckpt.fingerprint(denial, SYSTEM_PROMPT, self.model.model),
                      complete=True)
        except Exception as exc:
            state.log(f"could not close checkpoint ({type(exc).__name__})")

        if self.audit_log is not False:
            from audit import record_run
            record_run(state, self.model.model, self.audit_log)
        return state

    def _run(self, denial: DenialRecord) -> AgentState:
        state = AgentState(denial=denial)

        # Step 0: deterministic lookup, before the model is involved at all.
        category, meaning = self.code_lookup.lookup(denial.carc)
        state.code_category = category
        state.code_meaning = meaning
        state.log(f"code lookup -> {category or 'UNMAPPED'}")

        if category is None:
            if not self.web_lookup:
                state.decision = Decision.ESCALATE
                state.stop_reason = "unmapped_denial_code"
                state.log("guardrail -> escalate (unmapped_denial_code)")
                return state

            # Day 11. Let the model run so it can look the code up and hand a
            # human something better than "unknown code". Provenance is set to
            # web here, and the guardrail will refuse to authorise anything on
            # it regardless of what the model concludes.
            state.code_provenance = "web"
            state.log("code unmapped -> web lookup allowed, "
                      "outcome cannot exceed escalate")

        state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._initial_context(state)},
        ]

        # Day 9. Pick up work already paid for on this claim. A checkpoint
        # written against a different prompt, guardrail set or claim is
        # refused rather than resumed, because mixing two versions of the
        # agent inside one run would be worse than repeating the calls.
        import checkpoint as ckpt
        ckpt_key = ckpt.run_key(denial, self.model.model)
        ckpt_fp = ckpt.fingerprint(denial, SYSTEM_PROMPT, self.model.model)
        if self.resume:
            saved = ckpt.load(ckpt_key, ckpt_fp)
            if saved:
                ckpt.restore(state, saved)
                ckpt.mark_resumed(ckpt_key)

        while True:
            if state.steps_taken >= self.max_steps:
                state.decision = Decision.ESCALATE
                state.stop_reason = "step_limit_reached"
                state.log("step limit reached -> escalate")
                return state

            state.steps_taken += 1
            # Written before the call, so a process killed mid-request still
            # keeps everything up to this point.
            ckpt.save(state, ckpt_key, ckpt_fp)

            try:
                action = self.model.step(state.messages)
            except ValidationError as exc:
                state.decision = Decision.ESCALATE
                state.stop_reason = "model_output_failed_validation"
                state.log(f"invalid model output -> escalate ({exc.error_count()} errors)")
                return state

            if action.action == "call_tool":
                self._handle_tool_call(state, action)
                # A completed tool call is the work most worth not repeating.
                ckpt.save(state, ckpt_key, ckpt_fp)
                continue

            state.judgment = action
            state.log(
                f"model judged {action.proposed_decision} "
                f"(category={action.denial_category}, confidence={action.confidence})"
            )
            break

        decision, reason = validate_action(state)
        state.decision = decision
        state.stop_reason = reason
        state.log(f"guardrail -> {decision.value} ({reason})")

        if decision == Decision.APPEAL:
            self._draft_appeal(state)

        return state

    def _initial_context(self, state: AgentState) -> str:
        d = state.denial

        # Day 7. What happened to this claim, this member and this denial code
        # before now. Facts only, with no recommendation attached: a model told
        # "this was escalated last time" will otherwise read that as an
        # instruction to escalate again rather than as evidence to weigh.
        history = "Memory disabled for this run."
        if self.use_memory:
            try:
                from store import recall
                state.memory = recall(d.claim_id, d.patient_id, d.carc)
                history = state.memory.as_context()
            except Exception as exc:
                # A missing or unreadable store must not stop a claim being
                # worked. Degrade to no memory and say so in the trace.
                state.log(f"memory unavailable ({type(exc).__name__})")
                history = "Prior history could not be read."

        return f"""\
Claim ID: {d.claim_id}
Payer: {d.payer}
Denied amount: ${d.amount:,.2f}
CARC: {d.carc}  RARC: {d.rarc or "N/A"}

Standard meaning of this denial code:
{state.code_meaning}

Payer's stated explanation:
{d.payer_explanation}

Documentation on file:
{d.documentation_summary or "(none)"}

Prior history:
{history}
"""

    def _handle_tool_call(self, state: AgentState, action: ModelAction) -> None:
        tool = action.tool

        if tool is None or tool not in TOOLS:
            observation = f"Error: '{tool}' is not an available tool."
            state.log(f"model requested unknown tool '{tool}' -> refused")
        elif tool in UNMAPPED_ONLY_TOOLS and state.code_provenance == "table":
            # The code is known. A public search must not be usable to argue
            # against the curated definition.
            observation = (
                f"Error: {tool} is only available when the denial code is not "
                f"in the trusted table. This code is."
            )
            state.log(f"model requested '{tool}' on a mapped code -> refused")
        elif tool in state.tools_called:
            observation = f"Error: {tool} has already been called this run."
            state.log(f"model repeated tool '{tool}' -> refused")
        else:
            observation = TOOLS[tool](state.denial, state.code_category)
            state.tools_called.append(tool)
            state.observations.append(f"{tool}: {observation}")
            state.log(f"model called {tool} ({action.tool_reason or 'no reason given'})")

        state.messages.append(
            {"role": "assistant", "content": action.model_dump_json(exclude_none=True)}
        )
        state.messages.append(
            {"role": "user", "content": f"Tool result:\n{observation}"}
        )

    def _draft_appeal(self, state: AgentState) -> None:
        d = state.denial
        j = state.judgment
        assert j is not None

        observations = "\n".join(f"- {o}" for o in state.observations) or "- (none)"

        state.appeal_draft = f"""\
RE: Appeal of denied claim {d.claim_id}

Payer: {d.payer}
Denied amount: ${d.amount:,.2f}
CARC: {d.carc}   RARC: {d.rarc or "N/A"}

Stated denial reason:
{state.code_meaning}

Assessed root cause:
{j.root_cause}

Supporting information gathered:
{observations}

Documentation summary:
{d.documentation_summary}

NOTE: policy language here has not been verified against the payer's published
policy. Human review is required before submission.
"""
        state.log("appeal draft created")
