# Denial Triage & Appeal Drafting Agent

A 30-day build, in public, including what breaks.

## The problem

When a medical claim is denied, someone has to read the denial, work out what
actually went wrong, decide whether it is worth appealing, and draft the appeal.
That takes roughly 20–45 minutes per denial. A large share of valid denials are
never appealed at all, because nobody has the hours.

This is a document-in, judgment, document-out workflow — the shape an agent can
genuinely do, provided it knows when to stop.

Built on public CARC/RARC denial reason codes and synthetic claim data.

## What it does today

```
denial record
  -> deterministic CARC lookup
  -> model judgment (structured output)
  -> deterministic guardrail validation
  -> gather rationale / draft appeal / stop
```

The model owns judgment: denial category, root cause, proposed decision,
confidence, and whether a policy lookup is needed.

The deterministic layer owns liability: it reviews the proposed action and can
override it. The model can propose; only the guardrail layer authorizes.

## What it deliberately refuses to do

| Condition | Outcome |
|---|---|
| Denial code has no trusted mapping | Escalate before the model is called |
| Model's category disagrees with the code lookup | Escalate |
| Appeal proposed with no supporting documentation | Escalate |
| Category where the record alone can't justify an appeal | Escalate |
| Model confidence below floor | Escalate |
| No supporting rationale retrieved | Escalate |
| Step limit reached | Escalate |

It also never quotes payer policy language it has not retrieved. Every generated
draft states this explicitly and requires human review before submission.

**On confidence:** self-reported model confidence is not calibrated. It is used
as a soft input, never as the sole gate on a liability boundary. Whether it
predicts correctness at all is something the Day 18 evaluations will measure.

## Running it

```bash
pip install openai pydantic python-dotenv
```

`.env` (not committed):

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

```bash
python agent.py        # single case
python run_cases.py    # three cases across three paths
```

## Build log

- **Day 1** — Agent loop. Model judgment with structured output, deterministic
  guardrail layer, step limit, execution trace. Three verification cases:
  authorized appeal, guardrail refusal, unmapped-code escalation.

## Plan

| Week | Goal |
|---|---|
| 1 | Complete a real loop — tools, guardrails, memory, audit trail |
| 2 | Survive failure — structured outputs, checkpointing, resume mid-task |
| 3 | Be measurable — golden dataset, evaluations, cost per run |
| 4 | Be defensible — explain it to an engineer and to a non-technical executive |
