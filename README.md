# Denial Triage & Appeal Drafting Agent

A 30-day build, in public, including what breaks.

## The problem

A medical claim gets denied. Someone has to read the denial, work out what
actually went wrong, decide whether it's worth appealing, and write the appeal.
That's 20 to 45 minutes a claim. Denials that never get appealed are money the
provider simply doesn't collect.

It's a document-in, judgment, document-out job. The shape an agent can do,
as long as it knows when to stop.

Public CARC/RARC denial codes, synthetic claim data.

## What it does today

```
denial record
  -> deterministic CARC lookup (always first)
  -> prior history recalled from the store
  -> model turn
       |- requests a tool -> loop runs it, appends the result,
       |                     hands control back to the model
       `- returns judgment -> deterministic guardrail validation
  -> draft appeal / escalate / stop
  -> audit record written, always
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
| `search_denial_code` | Public web search for an unknown code. Offered only when the curated table has failed. Results are unverified and cannot support an appeal. |

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
| Model contradicts a decision a human already made | Escalate |
| Category came from a web search rather than the code table | Escalate, whatever the model proposed |
| Appeal proposed with no supporting documentation | Escalate |
| Category where the record alone can't justify an appeal | Escalate |
| Appeal proposed with nothing retrieved | Escalate |
| Model confidence below floor | Escalate |
| Step limit reached | Escalate |
| Model requests an unknown tool | Refused, and told why |
| Model repeats a tool it already called | Refused, and told why |

A run also checkpoints after every step, so a crash or a quota stop resumes
rather than repeating work already paid for.

It also never quotes payer policy language it hasn't actually retrieved. Every
generated draft says so and requires human review before submission.

## On confidence

The guardrail escalates when the model reports confidence below 0.6. That
threshold was picked by feel, so days 3 to 5 went looking for what the number
actually tracks. Three findings, all in the build log:

1. The score goes **down** as the agent reads more. Empty context produces
   0.95, repeatedly, across two models and two different claims. Retrieval is
   therefore a precondition for authorization, and confidence is only
   consulted afterwards.
2. Across every run recorded so far, the floor has never once been the rule
   that decided anything.
3. The score is stable within a single model call and unstable across a
   multi-turn run.

Confidence is a soft input here, never the only gate on a liability boundary.

## On evaluating this

Three layers, and they answer different questions.

*Did the right document come back at all.* `eval_retrieval.py`, 30 labelled
queries, recall@1 and recall@3. Runs offline, costs no quota, and catches the
class of bug that shipped on day 6.

*Does the model actually use what came back.* Context ablation: plant a document
that contradicts the model's prior and check whether the decision moves.
`probe_retrieval.py`. Observing successful runs cannot answer this, because
agreement makes "read it" and "ignored it" produce the same output.

*Does it agree with a human.* The `resolutions` table is the shape of this and
it is the layer that decides whether a system like this ships. Not yet built —
that is week 3, and it needs claims a human has actually worked.

All three have to be run repeatedly per case rather than once, because days 4,
5 and 8 each showed a single run does not tell you what the system does.

## Audit trail

Every run appends one JSON object to `runs/runs.jsonl`: the claim, the model,
which tools ran in order, the observations they returned, what the model
proposed and at what confidence, what was actually authorized, and which
specific rule decided it.

The hook lives in `run()` rather than in each calling script, so a run cannot
finish without leaving a record.

Every run also lands in SQLite. `claims` is the work queue, `runs` is the
queryable copy of the log, and `resolutions` holds what a human actually
decided — the only ground truth in the system, and the thing week 3's
evaluation has to score against. Scoring against the agent's own past output
would only measure whether it agrees with itself.

```bash
python audit.py           # summarise the raw JSONL log
python store.py stats     # the same thing as SQL, plus stability per claim
```

## Running it

```bash
pip install openai pydantic python-dotenv scikit-learn ddgs
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers   # optional; falls back to TF-IDF without it
```

`.env` (not committed):

```
LLM_API_KEY=your-key
LLM_MODEL=gemini-3.6-flash
```

Currently running on Google AI Studio's free Gemini tier through its
OpenAI-compatible endpoint. `LLM_BASE_URL` and `LLM_MODEL` are both read from
the environment, so any OpenAI-compatible provider works without touching the
code.

```bash
python agent.py                                       # single case
python run_cases.py                                   # five cases
python calibrate.py gemini-3.6-flash 10               # one call per run, no tools
python calibrate_tools.py gemini-3.7-flash 5 100046   # full agent, tools enabled
python audit.py                                       # read the run log
python build_corpus.py                                # write the policy corpus
python retrieval.py                                   # build index, test queries
python probe_retrieval.py gemini-3.7-flash 3          # is retrieval load-bearing
python store.py init                                  # create the database
python store.py stats                                 # what every run so far says
python store.py history CLM-100046                    # one claim's full history
python eval_retrieval.py --verbose                    # labelled retrieval set
python check_second_query.py                          # does the second query earn its place
python checkpoint.py list                             # what is currently resumable
python checkpoint.py sweep                            # drop completed checkpoints
python test_guardrails.py                             # every guardrail, no API calls
python websearch.py 204                               # look up an unmapped code
python websearch.py --cache                           # what has been looked up
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
runs, tools deliberately removed.

```
gemini-3.6-flash   appeal 10/10   confidence 0.95 every run   stdev 0.000
gemini-3.7-flash   appeal          confidence 0.95            (quota cut it short)
```

Two different models, identical answer, zero variance. So the model isn't the
variable, which was the day 2 hypothesis, and it was wrong.

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

*The multi-turn path isn't deterministic.* One call at temperature 0 returns
0.95 every time. Several calls in sequence return a spread, and the proposed
decision flips between runs. Four of the five runs called the same tools in the
same order and still disagreed with each other, so this isn't the agent taking
different routes. It compounds inside the conversation itself.

*So day 3's "0.75 with tools" was one draw, not a stable value.* The no-tools
side of that comparison holds up. The with-tools side was noisier than the
write-up implied.

*The confidence floor never fired.* Not on a single run. Every escalation came
from the CARC 96 category rule. On the two runs where the model proposed an
appeal at 0.85, confidence would have let them straight through.

One guardrail change followed: an appeal can no longer be authorized from a
judgment made with nothing retrieved, regardless of the score. Evidence is a
precondition; confidence is a check applied after it.

**Day 5** — Audit trail. Every run now writes a structured record. Two reasons
it landed here: a human picking up an escalation needs to know which run they
are looking at, and day 4 showed the runs differ. And week 3 can't compute a
pass rate over runs that were never persisted.

Then five runs of CLM-100046 on gemini-3.7-flash. That case matters because
`authorization_missing` is not in `NEVER_AUTO_APPEAL`, so nothing deterministic
stands behind the model.

```
run 1   no tools     0.95   blocked
run 2   both tools   0.85   appeal drafted
run 3   no tools     0.95   blocked
run 4   no tools     0.95   blocked
run 5   rate limit, died
```

Three of the four completed runs called no tools at all. Nothing errored; the
agent simply didn't retrieve anything before deciding. So it isn't only the
judgment that varies between runs, it's whether the agent does any research at
all.

All three of those reported 0.95. Same value day 3 found on a different claim
and a different denial code with an empty context, which is the first time that
finding has reproduced on data it wasn't derived from.

All three were stopped by `appeal_proposed_without_retrieved_evidence`, the rule
added on day 4. Without it, three appeals get authorized on this claim off a
judgment made from reading nothing, and the confidence floor would not have
blocked any of them.

Open: why the agent retrieves on some runs and not others, given identical
input and settings. Unresolved.

**Day 6** — Real retrieval. `retrieve_policy` was an if-statement returning two
hardcoded sentences with invented policy numbers. Every result before today
rests on the agent "retrieving" from that stub, so nothing measured before day 6
is directly comparable to anything measured after.

Replaced with a corpus and a retriever: 14 documents, 61 chunks, chunked on the
documents' own section headings. CARC references use the real X12 code meanings;
payer policies are synthetic and labelled as such in every file. Embeddings run
locally through `all-MiniLM-L6-v2`, so retrieval costs nothing against the daily
API quota. A TF-IDF backend sits behind the same interface as a fallback.

Three retrieval bugs, all found by testing rather than reading the code.

*The category filter wasn't filtering.* Only the CARC documents declared a
category, so every policy document passed through untagged. A medical necessity
query returned the non-covered policy as its top hit, which is the exact
confusion this project exists to prevent.

*Documents without `##` headings became one chunk each.* The CARC references
were single long chunks, and length normalisation buried them under short
unrelated passages. A timely filing query ranked an appeal-format section on
"Representation" above the document about filing deadlines.

*The top k came back as k chunks of one document.* This was the important one.
A policy stating that this payer no longer accepts appeals on the
authorization-not-submitted basis could not reach the top three on a claim about
exactly that scenario, because all three slots were taken by one file. Fixed
with a per-document cap and a second query: the claim text finds documents about
what the claim *is*, and a separate decision-oriented query finds documents about
what can be *done* with it. Those two sets share almost no vocabulary.

Embeddings versus TF-IDF, measured on the same query: "claim filed after the
deadline" scored 0.113 lexically and 0.695 semantically. Lexical search cannot
match "deadline" to "time limit" or "filed" to "filing".

Then a test of whether the model actually reads what comes back: inject one
policy passage saying the appeal route has been withdrawn, and see whether the
decision moves off its recorded baseline of appeal at 0.85. The passage now
retrieves correctly. The runs died on the daily quota before producing an
answer, so this is open.

Open: whether retrieval is load-bearing or decoration. Every case in the full
run landed on the same decision, with the same confidence, as it did with the
stub. That is either because the corpus says what the stub said, or because the
model is going on the denial category and ignoring the retrieved text. Not yet
distinguishable.

Also open: the top-ranked passage on the authorization case is a general
requirements section rather than the one describing the claim's actual
situation. Retrieval is finding the right document and the wrong part of it.
Reranking is the next step.

**Day 7** — Memory, on SQLite. Two problems closed. The claims were hardcoded
Python objects inside the scripts, and the agent had no memory between runs at
all — which is survivable when a run is deterministic and dangerous when it
isn't.

Three tables. `claims` is the work queue. `runs` replaces `runs.jsonl` as the
queryable copy, with the JSONL kept as the append-only raw log. `resolutions`
records what a human actually decided, which is the only ground truth in the
system — an agent's own past output is not.

Memory here is a lookup, not a vector store. The useful questions are exact:
has this claim been seen, has this member been through this before, what
normally happens with this denial code. Those are joins, and pretending
otherwise would be architecture for its own sake.

Prior history is rendered into the prompt as facts with no recommendation
attached, plus a prompt rule stating that history is context and not
instruction. Without that, a model shown "escalated last time" tends to read it
as an instruction rather than as evidence, and memory becomes a feedback loop.

One new guardrail follows from having memory at all: where a human has already
resolved a claim and the model proposes something different, the run escalates
rather than quietly overriding them. That rule could not exist before today.

Memory fails soft. An unreadable store logs the failure and the claim still
gets worked.

What twenty runs across five claims now say, as one query rather than an
impression:

```
what decided each run
  4  appeal_authorized:evidence=2
  3  appeal_authorized:evidence=1
  3  appeal_proposed_without_retrieved_evidence
  3  denial_appears_correct_on_record
  3  model_requested_escalation
  3  unmapped_denial_code

confidence floor fired on 0 of 19 runs
```

*The confidence floor has never once been the deciding rule.* Nineteen runs,
three models, two prompt versions, a stubbed retrieval layer and a real one. It
is not a threshold, it is decoration. Six other rules split the work fairly
evenly between them.

*Four of five claims are stable. The unstable one is the only claim no
deterministic rule covers.* CLM-100046 has returned two distinct outcomes across
seven runs, and `authorization_missing` is the one category not in
`NEVER_AUTO_APPEAL`. Every claim the rules stand behind is stable. The one they
don't is the one that moves.

Also worth recording: adding real retrieval on day 6 and memory on day 7 changed
no decision on any case. Two substantial additions to what the model can see,
and not one outcome moved. That points at the day 6 open question rather than
answering it — the model may be deciding largely from the deterministic category
lookup, with everything else as decoration.

**Day 8** — Answered the day 6 question, and was wrong about it first.

The observational case for "retrieval is decoration" looked strong. CLM-100046,
run on three separate days under three retrieval implementations: on 27 Aug the
stub returned nothing at all for this denial category, on 28 Aug real retrieval
returned the governing policy section, on 31 Aug better ranking returned a more
precise passage. Appeal at 0.85 all three times. Nineteen logged runs behind it.

That inference is invalid, and the flaw is structural rather than incidental.
Every document retrieved across those runs agreed with what the model would
have concluded anyway. Under agreement, "consulted the evidence" and "ignored
the evidence" produce identical output. The two hypotheses are indistinguishable
by construction, and no volume of successful runs separates them.

Only contradiction discriminates. `probe_retrieval.py` plants one policy stating
this payer no longer accepts appeals where an authorization existed but was
omitted from the claim, then runs the case that policy governs.

```
baseline        appeal at 0.85, three days, two retrieval implementations
with amendment  escalate, do_not_appeal, do_not_appeal
decision moved  3 of 3
cited the new policy in reasoning  3 of 3, one by document number
```

Retrieval is load-bearing. Nineteen observational runs supported a false
conclusion; three adversarial runs corrected it in four minutes.

The three runs split escalate / do_not_appeal / do_not_appeal, so retrieval
being load-bearing and the run-to-run instability from days 4 and 5 are both
true at once.

*Retrieval evaluation.* `eval_retrieval.py`, 30 queries each labelled with the
document that must come back. The labels are legitimate rather than invented —
this corpus was written for this project, so which document answers which
question is known. Includes a trap case worded like a medical necessity
question but filed under `noncovered_charge`, because that confusion is the
mistake the whole project exists to prevent.

It found a live defect on its first run. General-guidance chunks tagged
`Category: any` were being down-weighted unconditionally, including on queries
with no category filter — where the query is itself general and nothing else
can answer it. Three appeal-format queries were missing the appeal-format
policy. The penalty now applies only when a category is named.

```
                    recall@1   recall@3
tfidf                    80%       100%
embeddings               97%       100%
```

*Re-examined the day 6 second query.* Against the labelled set it costs 3%
recall@1 under embeddings and buys nothing: 100% and 100% without it. It was
compensating for lexical search being unable to connect a claim's facts to a
policy about appeal rights, and embeddings bridge most of that gap natively.

But `check_second_query.py` tests the one case the eval doesn't cover: the
planted contradiction, phrased entirely in appeal-process language, retrieved
against a claim described in clinical and administrative terms. It does not
surface without the second query — under lexical *or* semantic retrieval. The
vocabulary gap survives the move to embeddings.

So the second query stays, and the tradeoff is now measured rather than
assumed. It costs precision on documents that corroborate the claim and buys
recall on documents that dispute it. Only one of those two failure modes files
a bad appeal.

*Split the retry paths.* `RateLimitError` and `InternalServerError` were caught
in one block retrying six times. One upstream 500 was therefore retried five
times at 2, 4, 8, 16 and 32 seconds — five requests against a twenty-per-day
budget, returning nothing, which then made the next two runs fail on quota. A
429 is time-based and worth waiting out; a 500 means the upstream is unhealthy
and retrying does not make it healthy. Rate limits now retry up to four times
honouring the `retryDelay` the API actually sends (it asked for 57s on a day
the backoff capped at 32), server errors retry twice then give up.

**Day 9** — Checkpoint and resume. A run that died partway through lost
everything, and that has cost real requests twice this week. On 27 Aug a run
made two successful tool calls and then hit the rate limit; all of it was
discarded. On 1 Sep an upstream fault burned five of twenty daily requests and
returned nothing, which made the next two runs fail on quota. Every discarded
step was a request already paid for.

State is now written after each step is counted and again after each tool
result lands, since a completed tool call is the work most worth not repeating.
`run()` closes the checkpoint once the guardrails have decided, so a rerun
cannot reopen a settled case.

Three things it has to get right.

*A checkpoint must not outlive the code that wrote it.* Each one carries a
fingerprint of the claim, the system prompt, the confidence floor and the
never-auto-appeal set. A mismatch refuses the resume and starts clean, because
stitching two versions of the agent into one decision is worse than paying for
the calls twice.

*A conversation belongs to one model.* The run key is claim plus model, so a
half-finished run on one model is never handed to another.

*Resuming is not re-running.* The model sees the same rebuilt history, but days
4, 5 and 8 all showed the multi-turn path is not deterministic at temperature 0.
A resumed run may land somewhere an uninterrupted one would not, and the
checkpoint records that it was resumed so nobody later reads the result as a
clean run.

It got tested harder than planned. The first attempt died on an unhandled 503
at step 2, which is a better test than a clean interrupt: it proves the save
happened before the thing that killed the process, not during a graceful
shutdown. The resume picked it up, advanced to step 4, and stopped again when
the daily quota ran out. The next day's quota finished it.

```
CLM-100046:gemini-3.6-flash   done  (was resumed)
CLM-100046:gemini-3.7-flash   done
```

Three sessions, two crashes, two days, and no step paid for twice. The final
result carries the resumed flag, so nobody later reads it as a clean run.

*Daily quota is not a retryable condition.* The same 429 covers per-minute
throttling and a per-day cap, and the retry logic was waiting 33, 58, 58 and 59
seconds on a limit that resets tomorrow. Google names which one it is in
`quotaId`, so a `PerDay` violation now raises immediately instead of spending
three minutes and four requests confirming the cap is still there.

*Also observed, unresolved.* The 3.7 run on this claim called one tool and
stopped, finishing at `evidence=1`. Every earlier run called both and finished
at `evidence=2`, and day 5 recorded runs that called neither. Three distinct
tool-use behaviours on one claim. The guardrail requires that *something* was
retrieved, not that everything relevant was, so an agent that checks the policy
but never verifies the authorization exists is authorised on the same footing as
one that does both. "Did it retrieve enough" is a different rule from "did it
retrieve anything," and only the second one exists.

**Day 10** — Tested the guardrails that had never fired.

Nineteen runs, six rules decided them. There are twelve rules. The other six
had never executed once: malformed model output, step limit, category
disagreement, missing documentation, confidence below the floor, and
contradicting a recorded human decision.

Six rules that have never run are six rules that might not work, and they are
the ones that matter most. Each exists to stop something, and the moment you
find out a stop does not stop is the moment it was needed.

They are unreachable with real calls. You would have to wait for a model to
misbehave in a specific way, on a claim shaped to expose it, on a budget of
twenty attempts a day. So `test_guardrails.py` drives the agent with a scripted
model that returns exactly what each case needs. Eleven checks, no API calls,
about a second.

Every test runs with audit, memory and resume disabled, because a test must not
write to the run log, read history that changes between runs, or resume a
checkpoint left by an earlier test.

Two of the checks are worth naming.

*The confidence floor is tested at its boundary, not just inside its range.*
One case at 0.6 exactly and one just below. A threshold tested only in the
middle of its range has an untested edge, and thresholds fail at the edge.

*The unmapped-code case asserts the model was never called at all.* The README
has claimed since day 1 that the deterministic lookup runs before the model is
involved. Nothing had ever confirmed it. Now a failed assertion would.

All eleven passed, which was not the expected result. Rules that have never
executed usually have something wrong with them. The finding is smaller than a
bug and more useful to be able to state: before today "the guardrails handle
malformed output" was an assumption, and now it is a check that runs before
every commit.

**Day 11** — Web search, and what to do with information you cannot trust.

`DenialCodeLookup` holds five CARC codes. There are hundreds and X12 revises
them, so any claim carrying a code outside that table escalated immediately.
Safe and useless: four of nineteen runs ended that way, joint most common
outcome in the log. At volume it means a human opens every claim the table has
not been updated for.

The agent can now look a code up. The interesting part is refusing to pretend a
definition found on the open web is the same kind of thing as the curated
table.

Three rules, enforced rather than hoped for.

*The tool does not exist when the table has the code.* Requesting it on a
mapped code is refused, so a public search can never be used to argue against a
definition that was deliberately curated.

*Everything it returns is labelled unverified inside the observation text*,
beside the content rather than in a prompt preamble, because that is where the
model actually reads.

*A web-derived category can never authorise an appeal.* It produces a better
escalation instead. A human still opens the claim, but with "the agent searched,
read this as a non-covered charge, and here is the policy it then pulled"
rather than "unknown code".

Results cache in SQLite for 30 days. Code definitions are stable, and a free
search endpoint deserves not to be hammered.

The end-to-end run on CARC 204:

```
[step 0] code lookup -> UNMAPPED
[step 0] code unmapped -> web lookup allowed, outcome cannot exceed escalate
[step 1] model called search_denial_code
[step 2] model called retrieve_policy
[step 3] model judged do_not_appeal (noncovered_charge, confidence 0.95)
[step 3] guardrail -> escalate (unverified_code_definition:searched_web,
                                model_read_it_as=noncovered_charge)
```

The model was right. CARC 204 is a benefit-plan exclusion, and `do_not_appeal`
was the correct call. The guardrail escalated anyway.

*Whether that is the right trade is an open question, not a settled one.* The
argument for it: the guardrail cannot tell a correct web-derived conclusion
from an incorrect one, and a rule that trusts unverified sources only when they
agree with the safe answer is harder to defend to a compliance reviewer than
one that treats provenance uniformly. The argument against: closing a claim
costs nothing and filing a bad appeal costs money, so the two directions are
not symmetric and the rule is currently generating human review it may not need.

Two bugs found while building it, both the same shape as the problem the day
was about — a fixed list that does not cover reality.

*Source ranking was a substring match.* `medicaid-documents.dhhs.utah.gov`, a
state Medicaid agency publishing an actual CARC table, was labelled unverified
alongside two billing blogs, because "hhs.gov" does not appear in
"dhhs.utah.gov". Now matched on domain suffix, which also rejects
`notx12.org.evil.com` — something a substring check would have accepted as
authoritative.

*Ranking was applied at fetch time, not read time.* Fixing the ranking rule did
not reach anything already cached. Sorting is now a read-time decision and the
cache holds raw search order.

*A note on the sources.* X12 maintains the code list but does not publish it as
crawlable pages, so a general search returns secondary copies: state Medicaid
documents, Medicare contractor pages, billing vendors. Four sources agreeing is
good evidence and is still not the standard, and a copy can be stale. A real
deployment would license the list from X12 or take it from a payer companion
guide. That is the reason everything here stays marked unverified.

The guardrail suite grew to 14, all offline. One existing test failed on the
first run after this change, which was correct: it encoded the old behaviour
where an unmapped code stopped before the model. Updated, plus three new cases
— the tool being refused on a mapped code, the appeal cap on web provenance,
and the old behaviour still holding when web lookup is disabled, since that is
what runs if the search dependency is missing.

## Plan

| Week | Goal |
|---|---|
| 1 | Complete a real loop: tools, guardrails, memory, audit trail |
| 2 | Survive failure: structured outputs, checkpointing, resume mid-task |
| 3 | Be measurable: golden dataset, evaluations, cost per run |
| 4 | Be defensible: explain it to an engineer and to a non-technical executive |

Week 3 has already changed shape. If one run can flip the decision, an
evaluation that scores each case once is measuring noise. Each case needs
repeated runs and a pass rate.

## Honest caveat

The deterministic layer still decides most cases. Day 4 showed it decided every
case on the one claim measured, and day 5 showed it catching three of four runs
on another. Whether the model contributes real signal, or just agrees with a
lookup table, is what week 3's evaluation set exists to find out.

Sample sizes here are small: four to five completed runs per case, one or two
models, a handful of claims. Enough to change how the system is built. Not
enough to claim any of it generalises.
