# Denial Triage & Appeal Drafting Agent

A 30-day build, in public, including what breaks.

## The problem

A medical claim gets denied. Someone has to read the denial, work out what
actually went wrong, decide whether it's worth appealing, and write the appeal.
That's 20–45 minutes a claim. Denials that never get appealed are money the
provider simply doesn't collect.

It's a document-in, judgment, document-out job — the shape an agent can do,
as long as it knows when to stop.

Public CARC/RARC denial codes, synthetic claim data.

## What it does today

```
denial record
  -> deterministic CARC lookup (always first)
  -> model turn
       |- requests a tool -> loop runs it, appends the result,
       |                     hands control back to the model
       `- returns judgment -> deterministic guardrail validation
  -> draft appeal / escalate / stop
```

The model handles judgment: what category the denial falls into, the likely
root cause, whether it looks appealable, how confident it is, and what it needs
to look up before deciding.

The deterministic layer handles liability. It reviews what the model proposed
and can refuse it. **The model proposes. Only the guardrails authorize.**

That isn't theoretical. Given a non-covered charge with an approved prior
authorization on file, the model decided the denial was wrong and proposed an
appeal at 0.98 confidence. The guardrail refused, because no amount of
documentation changes whether a service sits inside a benefit plan. The case
went to a human.

## Tools

The model gets turns. Each turn it either asks for a tool or gives its
judgment. When it asks, the loop runs the tool, appends the result to the
conversation, and calls the model again.

| Tool | What it returns |
|---|---|
| `retrieve_policy` | Payer policy statements for this denial category |
| `check_prior_authorization` | Whether the claim record references a prior auth |

The loop refuses unknown tool names and repeat calls to the same tool, and
tells the model why it was refused. A step limit bounds the whole thing.

The CARC lookup is deliberately *not* a tool. It's cheap, deterministic and
always useful, so it runs before the model is involved at all. Tool choice is
reserved for calls that are genuinely optional.

## What it refuses to do

| Condition | Outcome |
|---|---|
| Denial code has no trusted mapping | Escalate before the model is even called |
| Model's category disagrees with the code lookup | Escalate |
| Appeal proposed with no supporting documentation | Escalate |
| Category where the record alone can't justify an appeal | Escalate |
| Model confidence below floor | Escalate |
| Step limit reached | Escalate |
| Model requests an unknown tool | Refused, and told why |
| Model repeats a tool it already called | Refused, and told why |

It also never quotes payer policy language it hasn't actually retrieved. Every
generated draft says so and requires human review before submission.

## On confidence

The guardrail escalates when the model reports confidence below 0.6. That
threshold was picked by feel, so day 3 went looking for what the number
actually tracks. See the build log below — the short version is that it tracks
how much the agent has read, not how clear the case is.

Confidence is a soft input here, never the only gate on a liability boundary.

## Running it

```bash
pip install openai pydantic python-dotenv
```

`.env` (not committed):

```
LLM_API_KEY=your-key
LLM_MODEL=gemini-3.6-flash
```

Currently running on Google AI Studio's free Gemini tier through its
OpenAI-compatible endpoint. `LLM_BASE_URL` and `LLM_MODEL` are both read from
the environment, so any OpenAI-compatible provider — OpenAI, Groq, OpenRouter,
a local Ollama model — works without touching the code.

```bash
python agent.py                          # single case
python run_cases.py                      # five cases
python calibrate.py gemini-3.6-flash 10  # calibration harness
```

## Build log

**Day 1** — Agent loop. Model judgment with structured output, deterministic
guardrail layer, step limit, execution trace. Four verification cases:
authorized appeal, model-requested escalation, unmapped-code escalation, and a
guardrail override of a 0.98-confidence appeal.

Hit repeatedly by 503s from the free tier mid-run, which killed the whole run
and threw away work that had already succeeded.

**Day 2** — Tool use. The model requests tools and the loop executes them,
handing control back with the result. Two tools, plus refusals for unknown and
repeated calls.

Pulled exponential backoff forward from week 3. The free tier rate limits at
five requests a minute, and a five-case run went straight past it.

The interesting result: on a plain non-covered denial the model retrieved the
policy, read that clinical documentation doesn't create coverage, and changed
its own judgment from appeal to do-not-appeal. Tools changed what the model
concluded, not just how it explained itself.

**Day 3** — Calibration harness. Same case, same prompt, temperature 0, ten
runs, tools deliberately removed. Records decision and confidence, writes the
raw results to JSON.

```
gemini-3.6-flash   appeal 10/10   confidence 0.95 every run   stdev 0.000
gemini-3.7-flash   appeal          confidence 0.95            (quota cut it short)
```

Two different models, identical answer, zero variance. So the model isn't the
variable — which was the day 2 hypothesis, and it was wrong.

The only remaining difference is the tools. The run that returned 0.75 on day 2
had the policy lookup available. It read that documentation doesn't create
coverage, and got *less* sure. The runs above had nothing to read and were
certain.

So the confidence number is tracking how much the agent has read, not how clear
the case is. A better-informed agent reports lower confidence, which pushes it
toward escalation. The 0.6 floor knows none of this.

Still missing: ten runs *with* tools, to confirm 0.75 is stable rather than a
single draw.

Also learned that twenty requests per day per model is a real methodology
constraint. Measuring the agent costs more than running it.

## Plan

| Week | Goal |
|---|---|
| 1 | Complete a real loop — tools, guardrails, memory, audit trail |
| 2 | Survive failure — structured outputs, checkpointing, resume mid-task |
| 3 | Be measurable — golden dataset, evaluations, cost per run |
| 4 | Be defensible — explain it to an engineer and to a non-technical executive |

## Honest caveat

The deterministic layer still decides most cases. Whether the model is
contributing real signal, or just agreeing with a lookup table, is what week 3's
evaluation set exists to find out. Day 3 turned one guardrail input — the
confidence floor — from an assumption into a measured question. The rest of the
rules haven't been tested that way yet.
