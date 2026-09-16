"""
Day 18: what a decision costs.

Week 3's checkpoint is "an evaluated agent with known failure modes and
measured costs". The failure modes have been measured for a week. There is not
a single cost number anywhere in this project.

That gap matters more than it sounds, because week 4 is about defending the
system to somebody non-technical on pass rate, time saved, risk reduced, and
cost per claim. Three of those exist. The fourth does not.

Two decisions worth stating.

*Everything is priced as if it were paid.* The current providers are free
tiers, and nobody deploys claims software on a free tier. Pricing the work at
real published rates is the only number that means anything to somebody
deciding whether to run this.

*Cost is reported per outcome, not just per run.* A claim closed, an appeal
drafted, and an escalation are different products. An escalation that costs a
fraction of a cent and saves a reviewer twenty minutes is the business case;
"average cost per run" hides it.

Rates below are published per-million-token prices and go stale. Check them
before quoting any figure from this file.

    python costs.py rates        what each model is priced at
    python costs.py report       cost of everything recorded so far
    python costs.py report 45    ...with a reviewer valued at $45/hour
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass

from store import connect

# USD per million tokens, (input, output). Verified September 2026; these move.
# A model not listed here is priced at zero and flagged in the report rather
# than silently costing nothing.
RATES: dict[str, tuple[float, float]] = {
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.7-flash": (0.30, 2.50),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "qwen/qwen3.8-27b": (0.20, 0.60),
}

# What the agent is displacing. A denial takes a person 20 to 45 minutes; the
# low end is used deliberately, because overstating the saving is the easiest
# way to lose an argument with someone who does this work.
MINUTES_PER_CLAIM_MANUAL = 20
DEFAULT_REVIEWER_RATE = 40.0  # USD/hour

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    run_id             TEXT PRIMARY KEY,
    claim_id           TEXT NOT NULL,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    completion_tokens  INTEGER NOT NULL DEFAULT 0,
    api_calls          INTEGER NOT NULL DEFAULT 0,
    final_decision     TEXT,
    ts                 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_claim ON usage(claim_id);
"""


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.api_calls += 1


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate = RATES.get(model)
    if rate is None:
        return 0.0
    return (prompt_tokens / 1_000_000) * rate[0] + \
           (completion_tokens / 1_000_000) * rate[1]


def record(run_id: str, claim_id: str, provider: str, model: str,
           usage: Usage, final_decision: str | None, ts: str) -> None:
    init()
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO usage
               (run_id, claim_id, provider, model, prompt_tokens,
                completion_tokens, api_calls, final_decision, ts)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run_id, claim_id, provider, model, usage.prompt_tokens,
             usage.completion_tokens, usage.api_calls, final_decision, ts),
        )
        conn.commit()


# ------------------------------------------------------------------ reporting

def show_rates() -> None:
    print("USD per million tokens. Verified September 2026 and liable to move.\n")
    print(f"{'model':<26}{'input':>10}{'output':>10}")
    for model, (inp, out) in sorted(RATES.items()):
        print(f"{model:<26}{inp:>10.2f}{out:>10.2f}")
    print(f"\nA model missing from this table is costed at zero and flagged.")


def report(reviewer_rate: float = DEFAULT_REVIEWER_RATE) -> None:
    init()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM usage").fetchall()

    if not rows:
        print("No usage recorded yet. Run the agent once and try again.")
        return

    unpriced = sorted({r["model"] for r in rows if r["model"] not in RATES})

    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    total_calls = 0
    by_outcome: dict[str, list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    by_claim: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        c = cost_of(r["model"], r["prompt_tokens"], r["completion_tokens"])
        total_cost += c
        total_prompt += r["prompt_tokens"]
        total_completion += r["completion_tokens"]
        total_calls += r["api_calls"]
        by_outcome[r["final_decision"] or "unknown"].append(c)
        by_model[r["model"]].append(c)
        by_claim[r["claim_id"]].append(c)

    n = len(rows)
    print(f"{n} runs recorded, {total_calls} API calls\n")

    print(f"tokens in     {total_prompt:>12,}")
    print(f"tokens out    {total_completion:>12,}")
    print(f"total cost    ${total_cost:>11.4f}")
    print(f"per run       ${total_cost / n:>11.4f}")

    if unpriced:
        print(f"\n  NOT PRICED: {', '.join(unpriced)}")
        print(f"  Those runs are counted at zero. The totals above are a floor.")

    print("\ncost by outcome:")
    print(f"  {'outcome':<18}{'runs':>6}{'avg':>12}{'total':>12}")
    for outcome, costs in sorted(by_outcome.items(),
                                 key=lambda p: -len(p[1])):
        avg = sum(costs) / len(costs)
        print(f"  {outcome:<18}{len(costs):>6}${avg:>11.4f}${sum(costs):>11.4f}")

    print("\ncost by model:")
    for model, costs in sorted(by_model.items(), key=lambda p: -len(p[1])):
        avg = sum(costs) / len(costs)
        tag = "" if model in RATES else "   (not priced)"
        print(f"  {model:<26}{len(costs):>5} runs   ${avg:.4f} each{tag}")

    # ---- the comparison that matters
    per_run = total_cost / n
    manual_cost = reviewer_rate * (MINUTES_PER_CLAIM_MANUAL / 60)

    print("\n" + "=" * 62)
    print(f"A human working one denial: {MINUTES_PER_CLAIM_MANUAL} minutes at "
          f"${reviewer_rate:.0f}/hour = ${manual_cost:.2f}")
    print(f"The agent working one denial: ${per_run:.4f}")

    if per_run > 0:
        print(f"\nThe agent costs {manual_cost / per_run:,.0f}x less per claim.")
        print("\n  That ratio is the least interesting number here, and the")
        print("  most quotable, so it is worth saying what it leaves out.")
        print("  The agent does not replace the reviewer on every claim. It")
        print("  decides the ones the rules cover and hands the rest over.")

    # Escalations are the honest version of the business case.
    esc = by_outcome.get("escalate", [])
    if esc:
        esc_avg = sum(esc) / len(esc)
        print(f"\n  An escalation costs ${esc_avg:.4f} and still needs the "
              f"full {MINUTES_PER_CLAIM_MANUAL} minutes")
        print(f"  of human time. It saves nothing directly. What it buys is a")
        print(f"  reviewer opening the claim with the policy already pulled")
        print(f"  and the authorization already verified.")

    decided = [c for k, v in by_outcome.items()
               if k in ("appeal", "do_not_appeal") for c in v]
    if decided:
        share = len(decided) / n
        saved_minutes = len(decided) * MINUTES_PER_CLAIM_MANUAL
        saved_money = len(decided) * manual_cost - total_cost
        print(f"\n  {len(decided)} of {n} runs ({share:.0%}) reached a decision "
              f"without a human.")
        print(f"  At {MINUTES_PER_CLAIM_MANUAL} minutes each that is "
              f"{saved_minutes / 60:.1f} hours, "
              f"${saved_money:,.2f} net of API cost.")
        print(f"\n  Caveat that belongs next to that figure: day 17 measured "
              f"88% correct")
        print(f"  on a five-claim set. A wrong decision that nobody reviews "
              f"costs more")
        print(f"  than the twenty minutes it saved.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "rates":
        show_rates()
    else:
        rate = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REVIEWER_RATE
        report(rate)
