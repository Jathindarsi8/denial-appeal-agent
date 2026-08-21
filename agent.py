"""
Denial triage + appeal drafting agent.

Day 1: real agent loop.
    denial -> deterministic code lookup
           -> model judgment (structured)
           -> deterministic guardrail validation
           -> next action

The model owns judgment. The guardrail layer owns liability boundaries.
The model can propose an action; only the guardrail layer can authorize it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

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


class Decision(str, Enum):
    APPEAL = "appeal"
    DO_NOT_APPEAL = "do_not_appeal"
    ESCALATE = "escalate"


class Step(str, Enum):
    LOOKUP_CODE = "lookup_code"
    JUDGE = "judge"
    VALIDATE = "validate"
    GATHER_RATIONALE = "gather_rationale"
    DRAFT_APPEAL = "draft_appeal"
    STOP = "stop"


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


class DenialJudgment(BaseModel):
    """What the model is allowed to have an opinion about."""

    denial_category: DenialCategory
    root_cause: str = Field(description="One sentence: why this claim was denied.")
    proposed_decision: Literal["appeal", "do_not_appeal", "escalate"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str
    needs_policy_lookup: bool


@dataclass
class AgentState:
    denial: DenialRecord
    step: Step = Step.LOOKUP_CODE
    steps_taken: int = 0

    code_category: Optional[str] = None
    code_meaning: Optional[str] = None
    judgment: Optional[DenialJudgment] = None

    decision: Optional[Decision] = None
    rationale: list[str] = field(default_factory=list)
    appeal_draft: Optional[str] = None
    stop_reason: Optional[str] = None
    trace: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.trace.append(f"[step {self.steps_taken}] {message}")


# ---------------------------------------------------------------------- tools

class DenialCodeLookup:
    """Deterministic CARC lookup. Runs before the model so judgment has context."""

    CODES: dict[str, tuple[str, str]] = {
        "16": (
            "missing_or_invalid_information",
            "Claim/service lacks information or contains submission/billing errors.",
        ),
        "29": (
            "timely_filing",
            "The time limit for filing this claim has expired.",
        ),
        "50": (
            "medical_necessity",
            "Service was denied as not medically necessary by the payer.",
        ),
        "96": (
            "noncovered_charge",
            "Charge is considered non-covered under the payer's policy.",
        ),
        "197": (
            "authorization_missing",
            "Precertification, authorization, or notification was absent.",
        ),
    }

    def lookup(self, carc: str) -> tuple[Optional[str], str]:
        if carc in self.CODES:
            return self.CODES[carc]
        return None, f"No trusted local mapping is available for CARC {carc}."


class PolicyRetriever:
    """Day 2 replaces this with retrieval over real payer policy documents."""

    def retrieve(self, denial: DenialRecord, category: str) -> list[str]:
        if category == "medical_necessity":
            return [
                "Submitted documentation indicates the service was ordered by the treating clinician.",
                "The record contains clinical findings supporting the stated indication.",
            ]
        return []


# ------------------------------------------------------------ model judgment

JUDGE_SYSTEM_PROMPT = """\
You are a claims denial analyst. You read a denied medical claim and produce a
structured judgment.

You do not decide anything final. A deterministic validation layer reviews your
proposed decision and may override it. Your job is accurate assessment, not
advocacy.

Rules:
- Propose "appeal" only when the record contains concrete clinical or factual
  support that contradicts the stated denial reason.
- Propose "do_not_appeal" when the denial appears correct on the record.
- Propose "escalate" whenever the record is ambiguous, incomplete, or the
  decision depends on information you do not have.
- Never assert payer policy language. You have not read the payer's policy.
- Set confidence honestly. Low confidence is a useful signal, not a failure.
"""


class JudgeModel:
    def __init__(self, model: Optional[str] = None):
        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv(
                "LLM_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )
        self.model = model or os.getenv("LLM_MODEL", "gemini-3-flash")

    def judge(self, denial: DenialRecord, code_meaning: str) -> DenialJudgment:
        user_content = f"""\
Claim ID: {denial.claim_id}
Payer: {denial.payer}
Denied amount: ${denial.amount:,.2f}
CARC: {denial.carc}  RARC: {denial.rarc or "N/A"}

Standard meaning of this denial code:
{code_meaning}

Payer's stated explanation:
{denial.payer_explanation}

Documentation on file:
{denial.documentation_summary or "(none)"}

Respond with JSON only, matching this schema exactly:
{{
  "denial_category": one of ["medical_necessity", "missing_or_invalid_information",
      "noncovered_charge", "authorization_missing", "timely_filing",
      "duplicate_claim", "coding_error", "other"],
  "root_cause": "one sentence",
  "proposed_decision": one of ["appeal", "do_not_appeal", "escalate"],
  "confidence": number between 0.0 and 1.0,
  "reasoning_summary": "two or three sentences",
  "needs_policy_lookup": true or false
}}
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = completion.choices[0].message.content
        if raw is None:
            raise ValueError("Model returned no content")
        return DenialJudgment.model_validate_json(raw)

# ------------------------------------------------------------------ guardrails

CONFIDENCE_FLOOR = 0.6

# Categories where an appeal can never be authorized from the record alone.
NEVER_AUTO_APPEAL = {"noncovered_charge", "timely_filing", "duplicate_claim", "other"}


def validate_action(state: AgentState) -> tuple[Decision, str]:
    """
    The model proposed an action. Decide whether it is allowed to take it.

    Every rule here is a verifiable condition, not a vibe. Confidence is a soft
    input only -- self-reported confidence from a language model is not
    calibrated and must never be the sole gate on a liability boundary.
    """
    judgment = state.judgment
    assert judgment is not None

    if state.code_category is None:
        return Decision.ESCALATE, "unmapped_denial_code"

    if judgment.denial_category != state.code_category:
        return Decision.ESCALATE, (
            f"category_disagreement:code={state.code_category},"
            f"model={judgment.denial_category}"
        )

    if judgment.proposed_decision == "escalate":
        return Decision.ESCALATE, "model_requested_escalation"

    if judgment.proposed_decision == "do_not_appeal":
        return Decision.DO_NOT_APPEAL, "denial_appears_correct_on_record"

    # From here the model proposed an appeal.
    if judgment.denial_category in NEVER_AUTO_APPEAL:
        return Decision.ESCALATE, f"category_requires_human_review:{judgment.denial_category}"

    if not state.denial.documentation_summary.strip():
        return Decision.ESCALATE, "appeal_proposed_without_supporting_documentation"

    if judgment.confidence < CONFIDENCE_FLOOR:
        return Decision.ESCALATE, f"confidence_below_floor:{judgment.confidence:.2f}"

    return Decision.APPEAL, "appeal_authorized"


# ----------------------------------------------------------------- agent loop

class DenialAppealAgent:
    def __init__(
        self,
        code_lookup: DenialCodeLookup,
        judge: JudgeModel,
        policy_retriever: PolicyRetriever,
        max_steps: int = 6,
    ):
        self.code_lookup = code_lookup
        self.judge = judge
        self.policy_retriever = policy_retriever
        self.max_steps = max_steps

    def run(self, denial: DenialRecord) -> AgentState:
        state = AgentState(denial=denial)

        while state.step != Step.STOP:
            if state.steps_taken >= self.max_steps:
                state.decision = Decision.ESCALATE
                state.stop_reason = "step_limit_reached"
                state.step = Step.STOP
                break

            state.steps_taken += 1
            handler = {
                Step.LOOKUP_CODE: self._lookup_code,
                Step.JUDGE: self._judge,
                Step.VALIDATE: self._validate,
                Step.GATHER_RATIONALE: self._gather_rationale,
                Step.DRAFT_APPEAL: self._draft_appeal,
            }.get(state.step)

            if handler is None:
                state.decision = Decision.ESCALATE
                state.stop_reason = f"unexpected_step:{state.step}"
                state.step = Step.STOP
                break

            handler(state)

        return state

    def _lookup_code(self, state: AgentState) -> None:
        category, meaning = self.code_lookup.lookup(state.denial.carc)
        state.code_category = category
        state.code_meaning = meaning
        state.log(f"code lookup -> {category or 'UNMAPPED'}")
        state.step = Step.JUDGE

    def _judge(self, state: AgentState) -> None:
        state.judgment = self.judge.judge(state.denial, state.code_meaning or "")
        state.log(
            f"model proposed {state.judgment.proposed_decision} "
            f"(category={state.judgment.denial_category}, "
            f"confidence={state.judgment.confidence:.2f})"
        )
        state.step = Step.VALIDATE

    def _validate(self, state: AgentState) -> None:
        decision, reason = validate_action(state)
        state.decision = decision
        state.log(f"guardrail -> {decision.value} ({reason})")

        if decision != Decision.APPEAL:
            state.stop_reason = reason
            state.step = Step.STOP
            return

        state.step = (
            Step.GATHER_RATIONALE
            if state.judgment is not None and state.judgment.needs_policy_lookup
            else Step.DRAFT_APPEAL
        )

    def _gather_rationale(self, state: AgentState) -> None:
        state.rationale = self.policy_retriever.retrieve(
            state.denial, state.code_category or ""
        )
        state.log(f"retrieved {len(state.rationale)} supporting statements")

        if not state.rationale:
            state.decision = Decision.ESCALATE
            state.stop_reason = "no_supporting_rationale_retrieved"
            state.step = Step.STOP
            return

        state.step = Step.DRAFT_APPEAL

    def _draft_appeal(self, state: AgentState) -> None:
        denial = state.denial
        judgment = state.judgment
        assert judgment is not None
        bullets = "\n".join(f"- {item}" for item in state.rationale) or "- (none retrieved)"

        state.appeal_draft = f"""\
RE: Appeal of denied claim {denial.claim_id}

Payer: {denial.payer}
Denied amount: ${denial.amount:,.2f}
CARC: {denial.carc}   RARC: {denial.rarc or "N/A"}

Stated denial reason:
{state.code_meaning}

Assessed root cause:
{judgment.root_cause}

We request reconsideration on the following basis:

{bullets}

Documentation summary:
{denial.documentation_summary}

NOTE: This draft does not quote payer policy language. An authoritative policy
retrieval tool is not yet connected. Human review is required before submission.
"""
        state.stop_reason = "appeal_draft_created_requires_human_review"
        state.log("appeal draft created")
        state.step = Step.STOP


# ----------------------------------------------------------------------- demo

if __name__ == "__main__":
    denial = DenialRecord(
        claim_id="CLM-100042",
        patient_id="SYNTH-001",
        payer="Synthetic Health Plan",
        amount=1840.00,
        carc="50",
        rarc=None,
        payer_explanation="The payer states that the service was not medically necessary.",
        documentation_summary=(
            "The treating clinician documented persistent symptoms, prior conservative "
            "treatment failure, and the clinical indication for the ordered service."
        ),
    )

    agent = DenialAppealAgent(
        code_lookup=DenialCodeLookup(),
        judge=JudgeModel(),
        policy_retriever=PolicyRetriever(),
    )

    result = agent.run(denial)

    print(f"decision:    {result.decision.value if result.decision else None}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"steps:       {result.steps_taken}")
    print("\n--- TRACE ---")
    for line in result.trace:
        print(line)
    print("\n--- APPEAL DRAFT ---")
    print(result.appeal_draft or "(none)")
