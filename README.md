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
| **Appeal proposed with nothing retrieved** | **Escalate** |
| Model confidence below floor | Escalate |
| Step limit reached | Escalate |
| Model requests an unknown tool | Refused, and told why |
| Model repeats a tool it already called | Refused, and told why |

It also never quotes payer policy language it hasn't actually retrieved. Every
generated draft says so and requires human review before submission.

## On confidence

The guardrail escalates when the model reports confidence below 0.6. That
threshold was picked by feel, so days 3 and 4 went looking for what the number
actually tracks. Two findings, both in the build log:

1. The score goes **down** as the agent reads more. Empty context reads as
   certainty. So retrieval is now a precondition for authorization, and
   confidence is only consulted afterwards.
2. On the test case, the floor never fired once. Every escalation came from a
   deterministic category rule instead.

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
python agent.py                                # single case
python run_cases.py                            # five cases
python calibrate.py gemini-3.6-flash 10        # one call per run, no tools
python calibrate_tools.py gemini-3.6-flash 5   # full agent, tools enabled
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

The remaining difference was the tools. The run that returned 0.75 on day 2 had
the policy lookup available, read that documentation doesn't create coverage,
and got *less* sure. So the confidence number tracks how much the agent has
read, not how clear the case is.

Also learned that twenty requests per day per model is a real methodology
constraint. Measuring the agent costs more than running it.

**Day 4** — Measured the other path. `calibrate_tools.py` drives the real agent
rather than bypassing it, so what's measured is the production path including
guardrails. Five runs, tools enabled, same case and model as day 3.

```
proposed   escalate 3/5   appeal 2/5
confidence 0.85  0.75  0.80  0.85  0.85     stdev 0.045
final      escalate 5/5
```

Three things came out of that, and only the first was the one being looked for.

*The multi-turn path isn't deterministic.* One call at temperature 0 returns
0.95 every time. Several calls in sequence return a spread, and the proposed
decision flips between runs. Four of the five runs called the same tools in the
same order and still disagreed with each other, so this isn't the agent taking
different routes — it compounds inside the conversation itself.

*So day 3's "0.75 with tools" was one draw, not a stable value.* The no-tools
side of that comparison holds up. The with-tools side was noisier than the
write-up implied.

*The confidence floor never fired.* Not on a single run. Every escalation came
from the CARC 96 category rule — non-covered charges can't be auto-appealed,
full stop. On the two runs where the model proposed an appeal at 0.85,
confidence would have let them straight through. The thing actually holding the
line was a lookup table written on day 1.

One guardrail change followed: an appeal can no longer be authorized from a
judgment made with nothing retrieved, regardless of the score. Evidence is a
precondition; confidence is a check applied after it.

The consequence for week 3 is larger. If one run can flip the decision, then a
golden dataset scored once per case measures nothing. Each case needs repeated
runs and a pass *rate*.

## Plan

| Week | Goal |
|---|---|
| 1 | Complete a real loop — tools, guardrails, memory, audit trail |
| 2 | Survive failure — structured outputs, checkpointing, resume mid-task |
| 3 | Be measurable — golden dataset, evaluations, cost per run |
| 4 | Be defensible — explain it to an engineer and to a non-technical executive |

## Honest caveat

The deterministic layer still decides most cases — day 4 showed it decided
*every* case on the one claim that was measured. Whether the model contributes
real signal, or just agrees with a lookup table, is what week 3's evaluation set
exists to find out. Two guardrail inputs have now been turned from assumptions
into measured questions. The rest haven't.
