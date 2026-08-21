Denial Triage & Appeal Drafting Agent

A 30-day build, in public, including what breaks.

The problem

A medical claim gets denied. Someone has to read the denial, work out what actually went wrong, decide whether it's worth appealing, and write the appeal. That's 20–45 minutes a claim. Denials that never get appealed are money the provider simply doesn't collect.

It's a document-in, judgment, document-out job — the shape an agent can do, as long as it knows when to stop.

Public CARC/RARC denial codes, synthetic claim data.

What it does today
denial record
  -> deterministic CARC lookup
  -> model judgment (structured JSON, validated with Pydantic)
  -> deterministic guardrail validation
  -> gather rationale / draft appeal / stop

The model handles judgment: what category the denial falls into, the likely root cause, whether it looks appealable, how confident it is, and whether a policy lookup is needed.

The deterministic layer handles liability. It reviews what the model proposed and can refuse it. The model proposes. Only the guardrails authorize.

That isn't theoretical. In the fourth test case the model read a non-covered charge with an approved prior authorization on file, decided the denial was wrong, and proposed an appeal at 0.98 confidence. The guardrail refused, because no amount of documentation changes whether a service sits inside a benefit plan. The case went to a human.

What it refuses to do
Condition	Outcome
Denial code has no trusted mapping	Escalate before the model is even called
Model's category disagrees with the code lookup	Escalate
Appeal proposed with no supporting documentation	Escalate
Category where the record alone can't justify an appeal	Escalate
Model confidence below floor	Escalate
No supporting rationale retrieved	Escalate
Step limit reached	Escalate

It also never quotes payer policy language it hasn't actually retrieved. Every generated draft says so and requires human review before submission.

On confidence: a self-reported score from a language model isn't calibrated. It's a soft input here, never the only gate on a liability boundary. Whether it predicts correctness at all is something the Day 18 evaluations will measure.

Running it
bash
pip install openai pydantic python-dotenv

.env (not committed):

LLM_API_KEY=your-key
LLM_MODEL=gemini-3.5-flash

Currently running on Google AI Studio's free Gemini tier through its OpenAI-compatible endpoint. LLM_BASE_URL and LLM_MODEL are both read from the environment, so any OpenAI-compatible provider — OpenAI, Groq, OpenRouter, a local Ollama model — works without touching the code.

bash
python agent.py        # single case
python run_cases.py    # four cases across four paths
Build log

Day 1 — Agent loop. Model judgment with structured output, deterministic guardrail layer, step limit, execution trace. Four verification cases: authorized appeal, model-requested escalation, unmapped-code escalation, and a guardrail override of a 0.98-confidence appeal.

Also hit repeatedly by 503s from the free tier mid-run, which killed the whole run and threw away work that had already succeeded. Good early argument for the checkpointing in week 2 and the backoff in week 3.

Plan
Week	Goal
1	Complete a real loop — tools, guardrails, memory, audit trail
2	Survive failure — structured outputs, checkpointing, resume mid-task
3	Be measurable — golden dataset, evaluations, cost per run
4	Be defensible — explain it to an engineer and to a non-technical executive
Honest caveat

Across the four cases so far, the deterministic layer decided most of them. Whether the model is contributing real signal, or just agreeing with a lookup table, is exactly what week 3's evaluation set exists to find out. Numbers get posted either way.