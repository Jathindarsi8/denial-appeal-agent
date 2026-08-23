"""
Denial triage + appeal drafting agent.

Day 2: real tool use.

    denial
      -> deterministic CARC lookup (always first, never the model's choice)
      -> model turn
           |- requests a tool  -> loop runs it, appends the observation,
           |                      hands control back to the model
           `- returns judgment -> deterministic guardrail validation
      -> draft appeal / escalate / stop

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
from typing import Callable, Literal, Optional

from dotenv import load_dotenv
from openai import InternalServerError, OpenAI, RateLimitError
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

ToolName = Literal["retrieve_policy", "check_prior_authorization"]


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

    messages: list[dict] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    judgment: Optional[ModelAction] = None
    decision: Optional[Decision] = None
    appeal_draft: Optional[str] = None
    stop_reason: Optional[str] = None
    trace: list[str] = field(default_factory=list)

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
    """Stubbed policy retrieval. Real retrieval lands later in week 1."""
    if category == "medical_necessity":
        return (
            "Policy MN-04: services denied as not medically necessary may be "
            "appealed with clinical documentation showing the indication and "
            "prior conservative treatment. No exact policy text retrieved yet."
        )
    if category == "noncovered_charge":
        return (
            "Policy NC-11: non-covered services are excluded by the member's "
            "benefit plan. Clinical documentation does not create coverage. "
            "A benefit exception requires a separate written determination."
        )
    return "No policy statements found for this denial category."


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


TOOLS: dict[str, Callable[[DenialRecord, Optional[str]], str]] = {
    "retrieve_policy": retrieve_policy,
    "check_prior_authorization": check_prior_authorization,
}


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
        self.model = model or os.getenv("LLM_MODEL", "gemini-3.5-flash")

    def step(self, messages: list[dict]) -> ModelAction:
        """One model turn, with exponential backoff on rate limits and 503s."""
        delay = 2
        for attempt in range(6):
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
            except (RateLimitError, InternalServerError):
                if attempt == 5:
                    raise
                print(f"  ...upstream busy, waiting {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 32)
        raise RuntimeError("unreachable")


# ------------------------------------------------------------------ guardrails

CONFIDENCE_FLOOR = 0.6
NEVER_AUTO_APPEAL = {"noncovered_charge", "timely_filing", "duplicate_claim", "other"}


def validate_action(state: AgentState) -> tuple[Decision, str]:
    """Every rule here is a verifiable condition, not a vibe."""
    j = state.judgment
    assert j is not None

    if state.code_category is None:
        return Decision.ESCALATE, "unmapped_denial_code"

    if j.denial_category != state.code_category:
        return Decision.ESCALATE, (
            f"category_disagreement:code={state.code_category},model={j.denial_category}"
        )

    if j.proposed_decision == "escalate":
        return Decision.ESCALATE, "model_requested_escalation"

    if j.proposed_decision == "do_not_appeal":
        return Decision.DO_NOT_APPEAL, "denial_appears_correct_on_record"

    if j.denial_category in NEVER_AUTO_APPEAL:
        return Decision.ESCALATE, f"category_requires_human_review:{j.denial_category}"

    if not state.denial.documentation_summary.strip():
        return Decision.ESCALATE, "appeal_proposed_without_supporting_documentation"

    if j.confidence is None or j.confidence < CONFIDENCE_FLOOR:
        return Decision.ESCALATE, f"confidence_below_floor:{j.confidence}"

    return Decision.APPEAL, "appeal_authorized"


# ----------------------------------------------------------------- agent loop

class DenialAppealAgent:
    def __init__(self, code_lookup: DenialCodeLookup, model: ModelClient,
                 max_steps: int = 8):
        self.code_lookup = code_lookup
        self.model = model
        self.max_steps = max_steps

    def run(self, denial: DenialRecord) -> AgentState:
        state = AgentState(denial=denial)

        # Step 0: deterministic lookup, before the model is involved at all.
        category, meaning = self.code_lookup.lookup(denial.carc)
        state.code_category = category
        state.code_meaning = meaning
        state.log(f"code lookup -> {category or 'UNMAPPED'}")

        if category is None:
            state.decision = Decision.ESCALATE
            state.stop_reason = "unmapped_denial_code"
            state.log("guardrail -> escalate (unmapped_denial_code)")
            return state

        state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._initial_context(state)},
        ]

        while True:
            if state.steps_taken >= self.max_steps:
                state.decision = Decision.ESCALATE
                state.stop_reason = "step_limit_reached"
                state.log("step limit reached -> escalate")
                return state

            state.steps_taken += 1

            try:
                action = self.model.step(state.messages)
            except ValidationError as exc:
                state.decision = Decision.ESCALATE
                state.stop_reason = "model_output_failed_validation"
                state.log(f"invalid model output -> escalate ({exc.error_count()} errors)")
                return state

            if action.action == "call_tool":
                self._handle_tool_call(state, action)
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
"""

    def _handle_tool_call(self, state: AgentState, action: ModelAction) -> None:
        tool = action.tool

        if tool is None or tool not in TOOLS:
            observation = f"Error: '{tool}' is not an available tool."
            state.log(f"model requested unknown tool '{tool}' -> refused")
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