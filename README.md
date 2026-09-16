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
| `check_prior_authorization` | Verifies a referenced authorization against a system of record: existence, status, member, date window, procedure |
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
| Claim closed while verified evidence contradicts the denial | Escalate |
| Appeal proposed with no supporting documentation | Escalate |
| Category where the record alone can't justify an appeal | Escalate |
| Appeal proposed with nothing retrieved | Escalate (backstop; superseded below) |
| Required checks for the category were not run, either direction | Escalate |
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

*Does it agree with a human.* `golden.py` and `evaluate.py`, added on day 15.
Five labelled claims, every label determined by a published rule, scored as
pass@1 and pass^k rather than once per case.

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
python authorizations.py seed                         # populate the auth system
python authorizations.py PA-88213                     # check one authorization
python compare_providers.py --groq 3 --gemini 2       # where do providers disagree
python cases.py                                       # every claim and what the auth system says
python golden.py                                      # the labelled set and its justifications
python golden.py record                               # write it to the resolutions table
python evaluate.py groq 20                            # pass@1, pass^k, and which control decided
python run_cases.py groq                              # five cases on a named provider
python costs.py rates                                 # what each model is priced at
python costs.py report                                # cost per claim and per outcome
python probe_memory.py gemini-3.6-flash 3             # does it anchor on its own history
python probe_prompt.py openai/gpt-oss-120b 5 groq     # is one prompt line causing tool-skipping
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

**Day 12** — Tested whether the agent anchors on its own history, and made two
mistakes worth keeping in the record.

Day 7's memory renders prior outcomes into the prompt: "this claim has been
processed 6 times before, previous outcomes: appeal, escalate, appeal...". Those
are the agent's own outputs fed back to the agent. The prompt says prior history
is context, not instruction, and one run on one claim had not shifted. One run
on one claim is not evidence.

`probe_memory.py` runs three conditions with everything else identical: no
history, a fabricated history of six escalations, a fabricated history of six
appeals. The history is injected by replacing `store.recall` for the duration
and the run log is disabled, so the real log is never contaminated with runs
whose history was invented.

*Mistake one: the wrong baseline.* The first version ran on CLM-100046, which is
the one claim in the log with two distinct outcomes across seven runs, and the
only one no deterministic rule stands behind. Its own run-to-run variance is the
same size as the effect being measured, so no result could be attributed to the
history rather than the noise. Testing an intervention against your noisiest
case measures the noise.

*Mistake two: the analysis was willing to over-read a tie.* The escalate
condition returned escalate once and appeal once. The verdict logic used
`Counter.most_common`, which breaks a 1-1 tie by insertion order, and printed
ANCHORING. A coin flip was reported as a directional finding. A condition that
does not agree with itself is measuring variance, and it now reports
INCONCLUSIVE and says which conditions disagreed.

Both fixed: the probe now runs on CLM-100042, which returned appeal on every
real run, and a split is never reported as a direction.

The corrected version got one run through before the daily quota ran out, and
that run is the most interesting thing from the day. Escalate-history condition,
model proposed appeal at 0.90, and its reasoning ended:

> While previous reviews resulted in escalations, the retrieved policy confirms
> the present documentation meets all criteria for a successful first-level
> appeal.

It read the six escalations, named them, and overrode them citing retrieved
policy. Not ignored, not deferred to. Treated as a claim to be weighed against
evidence.

One run, so a hypothesis rather than a result, but a sharper one than the day
started with. The question is no longer "does it anchor" but "does it override
history when it has evidence and follow history when it does not". That suggests
a fourth condition: history that conflicts with the retrieved policy, versus
history with no policy available to check it against. If the mechanism is real,
it says exactly when this memory design is safe and when it is not.

**Day 13** — A second provider, and the guardrail gap it exposed within five
runs.

Added provider configuration first, for quota reasons: twenty requests a day on
the free Gemini tier had blocked work on three separate days. `ModelClient` now
takes a provider name, resolves its base URL and key from the environment, and
every run records which provider answered. Deliberately not using a pooling
library with automatic failover — this project measures run-to-run behaviour,
and a library that quietly fails over mid-experiment would put two runs from
different models in one condition with nothing in the record to show it. The
tool that solves the quota problem breaks the measurement.

The first real run on the second provider closed CLM-100046 at 0.96 confidence
having called `retrieve_policy` and nothing else. It never called
`check_prior_authorization`, so it never checked whether the authorization
existed — which is the entire basis for appealing that claim.

Two holes let that through.

*The evidence rule only applied to appeals.* It sat inside the appeal branch,
and `do_not_appeal` returned earlier, so a claim could be closed having read
nothing at all. `do_not_appeal` had been treated as the safe direction. It is
not. A wrongly filed appeal gets rejected and somebody notices; a wrongly closed
claim is money the provider never collects and there is nothing left to notice.

*And "did it retrieve anything" was the wrong test regardless.* That run had an
observation, just not the relevant one. This is day 9's `evidence=1` versus
`evidence=2` question, now with a cost attached.

Replaced with `REQUIRED_TOOLS`, a per-category set of checks that must have run
before either terminal decision. `authorization_missing` requires both the
policy lookup and the authorization check. Escalating requires nothing, because
escalating is not a conclusion — it is a handoff.

The documentation check moved above it: a claim with no documentation on file
cannot produce a worthwhile appeal however many tools were run, so that is the
more useful reason to report. Day 4's `appeal_proposed_without_retrieved_evidence`
is now unreachable, since every category requires at least the policy lookup. It
stays as a backstop for a category added later without a `REQUIRED_TOOLS` entry,
and is commented as such rather than quietly left in.

Four new tests, suite at 18. Then the fix caught the failure it was written for,
in a live run, hours later.

*What thirteen runs on the second provider looked like:*

```
required_checks_not_run   12
category_disagreement      1
```

Not one reached a valid decision. Three different rules stopped the first three
runs — nothing retrieved, one of two retrieved, and the model reading CARC 197
as `missing_or_invalid_information` when the table says `authorization_missing`.
That last rule was written on day 1, had never fired in a real run, and a second
provider triggered it within five.

Two things are true at once here, and the second matters. The model is not doing
the work. And a rule that stops everything is not a safety feature, it is a
shutdown — on this claim, with this provider, the agent currently cannot reach
any decision. Safe and useless is where day 11 started with unmapped codes, and
it was not accepted then either.

*Ruled out the prompt as the cause.* The system prompt has said since day 2:
"Call a tool only when the answer would actually change your assessment." The
claim notes already mention PA-77104, so a model could reasonably read that
instruction as making the authorization check redundant. `probe_prompt.py`
removes exactly that line and changes nothing else.

```
with the line     3 of 5 called retrieve_policy, 0 of 5 called the auth check
without the line  5 of 5 called retrieve_policy, 0 of 5 called the auth check
```

Removing it made the model *more* willing to use tools generally and made no
difference at all to the one that matters. Unanimous across five runs. The
instruction was never the cause.

The probe's own metric was wrong first, and in the same way day 12's was. It
counted runs that called no tools at all, and reported that skipping had
"dropped" while every single run was still missing the required check. Counting
bare tool use hides a run that called one of two.

So it is not reluctance to use tools. The reading at the time: this model does
not treat verifying a stated fact as necessary, because the notes already
mention PA-77104.

*Later the same day this reading was disproved, and the wrong inference is
left here because it is part of the record.* The explanation assumed the model was weighing whether
the lookup was worth making, and concluding not. It was not. Once
`check_prior_authorization` became a real verification that can contradict the
claim notes outright, the skip rate was unchanged at 5 of 5. The contents of
the tool have no bearing on whether it gets called, because the decision to
skip happens before the contents are known.

Which points at the fix, and it is a move already made once. On day 1 the CARC
lookup was judged too important to be a model choice, so it runs before the
model is involved. The same applies here: on an `authorization_missing` claim,
checking whether an authorization exists is not an optional lookup, it is the
question. Making the model elect to do it is a design error two different models
have now made. Next: run it as a step for the categories that require it, with
the result in context before the model's first turn.

*Also confirmed:* the five-case suite still behaves identically on the first
provider, so the stricter rule costs nothing where things were already working.
The checkpoint fingerprint refused to resume across the guardrail change, which
is the day 9 rule working on a real code change rather than a test.

**Day 13, continued** — Then the rule built that morning turned out to rest on
a tool that could not do its job.

```python
text = denial.documentation_summary.lower()
if "prior authorization" in text or "pa-" in text:
    return "A prior authorization reference appears in the claim documentation."
```

For twelve days this tool searched the notes the model had already read and
reported them back. It could not verify anything and it could not fail. The
rule written that morning required it before any decision on an
`authorization_missing` claim and wrote up the model's refusal to call it as a
fault of the model. The model was
right: skipping a lookup that returns a fact you already hold is correct
behaviour, and the rule was enforcing a ceremony.

Replaced with an authorization system of record, separate from the claim, that
the claim's own text cannot influence. It can now return things the notes never
could:

```
PA-99999   referenced in the notes, DOES NOT EXIST in the system of record
PA-88213   exists and is approved, but authorises procedure 64483 while
           29827 was billed. AU-07 treats a partial match as no authorization
PA-55010   revoked after issue, notes still reference it
PA-61200   expired before the date of service
PA-77104   exists, approved, in window, procedure matches
```

The first of those is the important one. A claim asserting an authorization
that was never issued used to produce "an authorization is referenced" and an
appeal filed on a lie.

`DenialRecord` gained optional `procedure_code` and `date_of_service`. Without
them the check can only confirm an authorization exists, not that it covers
this service, and it says so in its own output rather than staying quiet about
what it could not check. Three tests, suite at 21.

*Then the interesting part.* The morning's finding was that a second provider
skipped this check on 12 of 13 runs, and the explanation was that the tool was
useless.
Five runs after making it genuinely useful:

```
required_checks_not_run   5 of 5
```

Unchanged. The tool's contents have no bearing on whether it gets called. The
model reads "PA-77104 obtained before the date of service" in the notes, treats
that as settled, and never reaches the question of verifying it.

That is a cleaner finding than the one it replaced, and a worse one. The
distinction between a fact asserted in a document and a fact verified against a
system of record is the entire job in claims work. A model that does not appear
to represent that distinction cannot be given discretion over it.

*So the required checks now run before the model's first turn*, with their
results already in its context. Same conclusion as day 1's CARC lookup:
something required on every decision is not a tool, it is a step. The rule
stays as a backstop — the rule and the mechanism that satisfies it are separate
things, and a category added to `REQUIRED_TOOLS` but missing from `TOOLS` would
otherwise pass silently.

Worth naming what that leaves. Every category requires `retrieve_policy`, so on
a mapped code the model now has no optional tools at all. Tool choice was never
a real capability in this agent; it only looked like one, and two providers
disagreeing about when to use it is what made that visible.

**Day 14** — Where two providers disagree, and what memory does to a claim
nothing stands behind.

`compare_providers.py` runs every case on every configured provider several
times and separates three things that are easy to conflate: whether a provider
agrees with itself, whether the providers agree with each other, and which way
each one goes when they do not. Old runs in the log are not reused, because the
required checks now run before the model's first turn and the authorization
check is a real lookup — anything recorded earlier describes a different
system.

```
claim        groq (3 runs)        gemini (2 runs)
CLM-100042   appeal 3             appeal 2
CLM-100043   do_not_appeal 3      do_not_appeal 2
CLM-100044   escalate 3           escalate 2
CLM-100045   escalate 3           escalate 2
CLM-100046   appeal 3             appeal 2
```

Complete agreement, both providers, every case, every run. Which makes
yesterday's apparent provider disagreement on CLM-100046 something else
entirely.

*The confidence floor was finally under-run, and still did not fire.* CLM-100044
came back at 0.40 and 0.30 on the first provider — the first time in more than
twenty runs anything has dropped below 0.6. The unverified-code-definition rule
caught it first. The floor has now been under-run twice and has still never
been the deciding rule in any run.

*Then the memory effect.* The comparison script disables memory, because prior
history differs between providers and would contaminate the comparison. That
turned out to be the variable.

```
CLM-100046, one provider, 20 runs per condition, two sessions

              session 1        session 2
memory off    appeal  9/20     appeal  8/20      42% overall
memory on     appeal 16/20     appeal 14/20      75% overall
```

Eighty runs. The gap held across two independent sessions. Showing the agent a
claim's own prior outcomes roughly doubles the odds it appeals.

Three things have to be said alongside that.

*Without memory, this claim is a coin flip.* 17 of 40 appeals. Not "sometimes
unstable" — the agent has no opinion and is sampling, on a $1,375 decision. This
is the one case where no deterministic rule applies, and where the rules do not
carry a case there is nothing underneath.

*Memory moves the outcome and nothing here can say toward what.* The stored
history for this claim is mixed. Shown that mixture the model does not become
uncertain, it goes to 75% appeal. There is no ground truth to say whether that
is better than 42%. The `resolutions` table is still empty.

*And "memory on" is not a fixed condition.* The history accumulates as runs are
recorded, so this reproduces within a session and not across days. Yesterday's
memory-on runs on this claim pointed the other way, on a shorter history and a
much smaller sample. A clean version snapshots the history and runs both
conditions against that fixed state.

*Also worth recording: this claim has now fooled three separate analyses.* Day
12's anchoring probe read a 1-1 split on it as a directional finding. Yesterday
a 4-of-5 against a 3-of-3 looked like providers disagreeing. This morning a 3-2
against a 2-3 looked like a memory effect. All three were draws from the same
coin, and all three happened because the most interesting claim in the set is
also the noisiest one to measure anything on.

**Day 15** — Ground truth, a score, and finding out the previous score was
measuring an easier problem.

Every finding so far ended the same way: the agent does X on this fraction of
runs and nothing here can say whether X is right. The `resolutions` table had
been empty since day 7.

`golden.py` fills it, with one rule applied throughout: *a decision goes in the
set only if a published rule determines it.* Where the answer turns on clinical
judgment, the correct label is `escalate`, which is also what the system should
do. That is not a dodge. In claims work "a human decides this" is a real and
common correct answer, and a set that pretends otherwise measures the wrong
thing.

What the set therefore does not assert: whether a reviewer would agree with the
clinical merits. What it does assert is whether the record contains the elements
the payer's own policy requires, which is a documentation question.

```
CLM-100042  appeal         MN-04, the record contains all three required elements
CLM-100043  do_not_appeal  NC-11, documentation does not create coverage
CLM-100044  escalate       no trusted definition for the denial code
CLM-100045  escalate       NC-11, authorization conflicting with an exclusion
CLM-100046  appeal         AU-07, authorization verified against the system
```

*The honest caveat, stated here rather than left to be found.* MN-04, NC-11 and
AU-07 are synthetic documents written for this project. The golden set is
scored against a policy corpus by the same author as the labels. That does not
make the labels wrong, but it does mean the set measures internal consistency
with stated rules rather than agreement with a real payer.

Two of the labels were only defensible after day 13. Before the authorization
check became a real lookup, CLM-100046 rested on the claim notes asserting an
authorization existed.

*`evaluate.py` scores two numbers, and the gap between them is the point.*

```
pass@1   of all runs, what fraction were correct
pass^k   of the cases, what fraction were correct on EVERY run
```

A case right 3 times in 5 contributes 0.6 to the first and 0 to the second.
That is the correct treatment. An agent that files the right appeal most of the
time is not an agent that files the right appeal.

*Then the case set turned out to be two case sets.* CLM-100046 was defined in
`run_cases.py` and again in `calibrate_tools.py` with different fields. The
`run_cases` copy had no procedure code and no date of service, and the
authorization check uses both. The same claim ID produced different evidence
depending on which file a test imported from.

The first evaluation, run against the thin copy:

```
pass@1   96%   (24 of 25 runs)
pass^5   80%   (4 of 5 cases correct every run)
```

After consolidating every definition into `cases.py` and re-running the same
evaluation against the same labels:

```
pass@1   76%   (19 of 25 runs)
pass^5   60%   (3 of 5 cases correct every run)
```

*The 96% was never a real score.* Without a procedure code the authorization
check could only report that PA-88213 exists. With one it reports that PA-88213
authorises 64483 while 29827 was billed, and that AU-07 treats a partial match
as no authorization. The agent was answering an easier question and the
evaluation reported it as accuracy.

*The drop has a direction, and it is the uncomfortable one.* Both regressions
moved toward closing claims.

```
CLM-100045   4/5 correct -> 1/5   four runs closed a claim the rule sends to a human
CLM-100046   5/5 correct -> 3/5   two runs closed a claim with a fully verified
                                  authorization supporting the appeal
```

On the first the model read "this authorization does not cover the billed
procedure" and closed the claim instead of escalating. On the second it read
"this authorization covers everything" and closed the claim instead of
appealing. Opposite evidence, same drift. More verified evidence made the agent
worse in one consistent direction, and it is the direction day 13 identified as
the one nobody notices: a rejected appeal gets seen, a wrongly closed claim is
money that quietly never arrives.

This is the second measurement lost to something that looked fixed and was not.
The first was the memory condition drifting between sessions because the stored
history kept growing. Both had the same shape, and both were only visible
because two numbers that should have matched did not.

*Still open.* The 76% is five runs per case, and day 14 established that five
runs is where this project keeps getting fooled. It needs replication. Memory
is disabled during evaluation, because the golden answers now live in a table
the agent can read, so the agent has never been scored as it would actually
run. And only one provider has been evaluated.

**Day 16** — Scoring the decision hides who made it.

Day 15's evaluation reported whether the final decision matched the golden
label. That cannot distinguish "the model was right" from "the model was wrong
and a rule refused it". Those are different systems with the same score, and if
most of the score is rules rescuing bad proposals then the honest description of
this project is a rules engine with a model attached.

So every run is now classified by which control produced the outcome:

```
clean         model right, rules agreed
rescued       model wrong, rules caught it
leaked        model wrong, rules let it through
overblocked   model right, rules overruled it
both_wrong    model wrong, rules wrong differently
no_judgment   stopped before the model judged
```

Two of ARISE's four modes are structurally impossible here on a mapped denial
code. The required checks run before the model's first turn, so it cannot
bypass search and cannot fail to retrieve. What remains is what the model does
with evidence it already holds, and what the rules do about it.

Then twenty runs per case instead of five, because five is the sample size that
has produced a wrong conclusion three times in this project.

```
100 runs, five claims

pass@1    83%   (83 of 100 runs)
pass^20   60%   (3 of 5 cases correct on every run)

model proposed the correct answer     76 of 100
system produced the correct answer    83 of 100

  76  model right, rules agreed
   7  model wrong, rules caught it
  17  model wrong, RULES LET IT THROUGH
   0  model right, rules overruled it
```

*Seven points of the score are the guardrails.* The model alone is at 76%. The
rules refused what it proposed on seven runs and were right to. That is now a
measured quantity rather than a claim.

*Zero overblocking across a hundred runs.* Not one correct proposal was refused.
That settles the day 11 worry that the provenance rule was generating
escalations nobody needed, at least on this case set.

*And all seventeen failures are the same failure.* Every one is the model
proposing `do_not_appeal` and nothing stopping it.

```
CLM-100045   7 runs closed a claim the rule sends to a human
CLM-100046  10 runs closed a claim with a fully verified authorization
```

Not a single wrong appeal. Not one failure in the other direction.

Which makes this a missing rule rather than a model problem. Appeals are
checked hard: category allowed, required checks run, documentation present,
confidence above the floor. `do_not_appeal` returns on
`denial_appears_correct_on_record` with almost nothing checked. The asymmetry
named on day 13 is now the sole cause of every failure in the evaluation.

CLM-100046 came back 10 of 20 — an exact coin flip on a claim where the
authorization system confirms the authorization exists, is approved, covers the
billed procedure, and falls inside its window. On $1,375.

*Still open.* One provider. Gemini has never been evaluated, so it is not yet
known whether the one-directional failure is a property of this system or of
this model.

**Day 17** — The rule that was missing, and a metric that was lying.

Day 16 produced seventeen failures across a hundred runs and every one was the
same event: the model proposed `do_not_appeal` and nothing stopped it. Not a
single wrong appeal. Filing an appeal passed four checks; closing a claim
returned on `denial_appears_correct_on_record` with almost nothing verified. It
was the only unguarded exit in the system.

`contradicting_evidence()` closes it. A claim cannot be closed when verified
evidence contradicts the denial:

```
authorization_missing   an approved authorization exists, covers the billed
                        procedure, right member, in date  ->  cannot close
                        an authorization exists but covers something else
                        ->  cannot close, AU-07 needs a human
noncovered_charge       an approved authorization alongside a non-covered
                        denial  ->  cannot close, NC-11 routes to review
```

*The rule does not read the model's context.* Tool output is prose and prose
changes, so a rule that matches on it breaks quietly. The authorization store
is a local lookup, so the guardrail queries it directly and forms its own view.

*And it deliberately stays out of the way in two cases.* No authorization on
file at all, and an authorization the notes cite that the system never issued.
That second one matters: a claim asserting a fake authorization supports the
denial, so closing it is correct. Without a test for it the fix would trade one
measured failure for another. Suite at 26.

```
                    day 16      day 17
pass@1                 83%         88%
pass^20                60%         80%
model alone            76%         70%
CLM-100045          13/20       20/20
CLM-100046          10/20        8/20
```

*CLM-100045 is now carried by the rule.* Eighteen of twenty runs are the
guardrail refusing what the model proposed. The case scores 100% and the model
is wrong almost every time. Those are very different statements and only the
mode breakdown separates them.

*CLM-100046 scored worse and got better.* The failures changed shape: twelve
runs that used to close a claim with a fully verified authorization now
escalate it instead. Flat scoring calls that a regression. It is not the same
event at all. One ends with $1,375 never collected and nobody aware. The other
ends with the claim on a reviewer's desk.

Which was a flaw in the measurement, not the system, so failures are now
weighted by whether anyone finds out:

```
 88  correct
 12  wrong, but a human finds out
  0  wrong, and nobody finds out
```

*Zero silent failures across a hundred runs.* Yesterday twenty-nine runs closed
claims that should not have been closed, leaving no artifact for anyone to
notice. Today every remaining failure is an escalation.

That is the ceiling for a refuse-only guardrail layer, and it is worth being
explicit about why. A rule can refuse a proposal. It cannot promote one. Turning
a bad close into an escalation is the most a deterministic layer can do without
becoming the thing making claims decisions, which is what this architecture
exists to prevent. The remaining twelve are the model being unreliable on a
claim the rules cannot decide for it.

*Also worth recording:* the model-alone figure moved from 76% to 70% between
two hundred-run evaluations with no change to the model or the claims. Even at
n=100 a few points are noise.

**Day 18** — What a decision costs.

Week 3's checkpoint is "an evaluated agent with known failure modes and
measured costs". The failure modes had been measured for a week. There was not
a single cost figure anywhere in the project, and week 4 is about defending
this to somebody non-technical on pass rate, time saved, risk reduced, and cost
per claim. Three of those existed.

Every API response carries a usage object with prompt and completion token
counts, and it was being discarded on every call since day 1. It now
accumulates on the run state and lands in its own `usage` table, separate from
the run log so a cost report does not depend on the log's schema.

Two decisions worth stating.

*Everything is priced as if it were paid.* The current providers are free
tiers, and nobody deploys claims software on a free tier. The only number worth
quoting is the one at published rates. A model missing from the rate table is
costed at zero and flagged in the report rather than quietly making the totals
a lie.

*Cost is reported per outcome, not per run.* A claim closed, an appeal drafted
and an escalation are different products, and an average hides that.

```
8 runs, 8 API calls, 7,279 tokens in, 1,989 out

per claim                       $0.0004
  on openai/gpt-oss-120b        $0.0003
  on gemini-3.6-flash           $0.0006

by outcome
  escalate       4 runs   $0.0003 each
  appeal         3 runs   $0.0004 each
  do_not_appeal  1 run    $0.0003
```

A reviewer working one denial: twenty minutes at $40/hour is $13.33. The agent
is four ten-thousandths of a dollar. The ratio is roughly 36,000 to 1, and it
is simultaneously the most quotable number here and the least informative, so
the report prints the argument against it immediately underneath.

*What the ratio leaves out.* An escalation costs the same fractions of a cent
and still consumes the full twenty minutes of human time. It saves nothing
directly. What it buys is a reviewer opening the claim with the policy already
pulled and the authorization already verified against the system of record.
Half the runs reached a decision with no human, which on eight claims is 1.3
hours. Day 17 measured 88% correct on a five-claim set, and a wrong decision
nobody reviews costs considerably more than the twenty minutes it saved.

*Two bugs found on the way, both the same shape.*

`except Exception: pass` around the cost write hid its own failure for a full
run of five claims. Worse, it turned out the same silent handler had been
hiding that the day 7 SQLite write was missing from the file entirely — runs
had stopped reaching the store and nothing said so. Both handlers now print
what failed. A write that is allowed to fail should still be allowed to
complain.

The cause was two copies of `audit.py` drifting apart, which is exactly what
cost a measurement on day 15 when one claim was defined in two files. Same
failure, second time: two copies of one thing, and nothing pointing out they
had diverged.

*Also observed:* `contradicts_human_decision` fired in a real run for the first
time. It was written on day 7 and had only ever run in tests, because the
resolutions table was empty until day 15. It only works now because there is
ground truth to contradict — and it also means `run_cases.py` is no longer a
clean test, since the agent can read the answer key. `evaluate.py` disables
memory for that reason; `run_cases.py` does not.

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
