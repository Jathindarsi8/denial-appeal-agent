"""
Day 11: web search, and what to do with information you cannot trust.

`DenialCodeLookup` holds five CARC codes. There are hundreds, and X12 revises
them. Any claim carrying a code outside that table escalates immediately, which
is safe and useless: four of nineteen runs so far ended that way, and it is the
joint most common outcome in the log. At volume it means a human opens every
claim the table has not been updated for.

So the agent gets a way to look a code up. But a definition pulled off the open
web is not the same kind of thing as the curated table, and the interesting
part of today is refusing to pretend otherwise.

Three rules follow from that, and they are enforced rather than hoped for.

*The tool only exists when the table has failed.* If the code is in the table,
the web tool is not offered at all. Otherwise the model could use a billing
blog to argue against a definition that was deliberately curated.

*Everything it returns is labelled unverified, in the text the model reads.*
Not in a comment, not in the prompt preamble. In the observation itself, next
to the content.

*A category that came from the web can never authorise an appeal.* It can only
produce a better escalation. A human still opens the claim, but they open it
with "this appears to be CARC 204, non-covered, per three sources, unverified"
instead of "unknown code, good luck". That is the whole value: web search
improves the quality of the handoff, it does not remove the handoff.

Results are cached in SQLite. Code definitions are stable, so re-searching the
same code is wasted latency and wasted goodwill with a free search endpoint.

    pip install ddgs

    python websearch.py 204        look up one code
    python websearch.py --cache    what is cached
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from store import connect

CACHE_DAYS = 30
MAX_RESULTS = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS code_lookups (
    carc        TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    results     TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);
"""

# Sites whose CARC definitions carry more weight than a billing blog. Not a
# whitelist — everything from the web stays unverified — but the model is told
# which results came from where so it can weigh them.
#
# Day 11: this started as a substring match and immediately mislabelled
# medicaid-documents.dhhs.utah.gov, a state Medicaid agency publishing an
# actual CARC table, as an unverified source. "hhs.gov" does not appear in
# "dhhs.utah.gov". Same failure as the five-entry code table this tool exists
# to work around: a fixed list that does not cover reality. Any government or
# academic domain now qualifies on its suffix.
PREFERRED_DOMAINS = ("x12.org", "wpc-edi.com", "noridian.com",
                     "palmettogba.com", "cgsmedicare.com", "ama-assn.org")
PREFERRED_SUFFIXES = (".gov", ".mil", ".edu")


@dataclass
class SearchResult:
    title: str
    body: str
    url: str

    @property
    def domain(self) -> str:
        try:
            return self.url.split("/")[2].replace("www.", "")
        except Exception:
            return "unknown"

    @property
    def preferred(self) -> bool:
        d = self.domain.lower()
        if any(d == p or d.endswith("." + p) for p in PREFERRED_DOMAINS):
            return True
        return any(d.endswith(sfx) for sfx in PREFERRED_SUFFIXES)


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------------ caching

def _cached(carc: str) -> Optional[list[SearchResult]]:
    init()
    with connect() as conn:
        row = conn.execute(
            "SELECT results, fetched_at FROM code_lookups WHERE carc = ?",
            (carc,)).fetchone()
    if row is None:
        return None

    fetched = datetime.fromisoformat(row["fetched_at"])
    if datetime.now(timezone.utc) - fetched > timedelta(days=CACHE_DAYS):
        return None

    return [SearchResult(**r) for r in json.loads(row["results"])]


def _cache(carc: str, query: str, results: list[SearchResult]) -> None:
    init()
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO code_lookups
               (carc, query, results, fetched_at) VALUES (?,?,?,?)""",
            (carc, query,
             json.dumps([{"title": r.title, "body": r.body, "url": r.url}
                         for r in results]),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()


# ------------------------------------------------------------------ search

def search_code(carc: str, use_cache: bool = True) -> list[SearchResult]:
    """Look up what a denial code means. Cached, because code definitions are
    stable and a free search endpoint deserves not to be hammered."""
    if use_cache:
        hit = _cached(carc)
        if hit is not None:
            # Sort on read, not only on fetch. Ranking rules change; cached
            # rows do not. Sorting at write time meant a fix to how sources
            # are ranked did not reach anything already cached.
            hit.sort(key=lambda r: (not r.preferred,))
            return hit

    query = f"CARC {carc} claim adjustment reason code denial meaning"

    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError(
            "web search needs the ddgs package: pip install ddgs"
        )

    raw = DDGS().text(query, max_results=MAX_RESULTS)
    results = [
        SearchResult(
            title=r.get("title", ""),
            body=r.get("body", ""),
            url=r.get("href", r.get("url", "")),
        )
        for r in raw
    ]

    if results:
        _cache(carc, query, results)

    # Preferred sources first, so if the model only reads the top result it
    # reads the better one. Applied after caching so the cache holds raw
    # search order and the ranking stays a read-time decision.
    results.sort(key=lambda r: (not r.preferred,))
    return results


def format_for_agent(carc: str, results: list[SearchResult]) -> str:
    """What the tool hands back. The warning is inside the observation, beside
    the content, because that is the only place the model reliably reads."""
    if not results:
        return (
            f"No usable web results for CARC {carc}. The code remains "
            f"unidentified. Do not guess at its meaning."
        )

    blocks = []
    for r in results:
        tag = "recognised source" if r.preferred else "unverified source"
        body = r.body[:400]
        blocks.append(f"[{r.domain} — {tag}]\n{r.title}\n{body}")

    return (
        f"UNVERIFIED web results for CARC {carc}. This code is NOT in the "
        f"trusted local table. Anything below came from a public search and "
        f"has not been checked against the X12 standard.\n\n"
        f"You may use this to describe what the denial appears to be, so a "
        f"human reviewer starts with context rather than nothing. You may NOT "
        f"treat it as established, and an appeal cannot be authorised on it.\n\n"
        + "\n\n---\n\n".join(blocks)
    )


# ------------------------------------------------------------------ cli

def show_cache() -> None:
    init()
    with connect() as conn:
        rows = conn.execute(
            "SELECT carc, fetched_at, results FROM code_lookups "
            "ORDER BY fetched_at DESC").fetchall()
    if not rows:
        print("nothing cached")
        return
    print(f"{len(rows)} code(s) cached\n")
    for r in rows:
        n = len(json.loads(r["results"]))
        print(f"  CARC {r['carc']:<8} {n} result(s)   {r['fetched_at']}")


def main() -> None:
    if "--cache" in sys.argv:
        show_cache()
        return
    if len(sys.argv) < 2:
        print(__doc__)
        return

    carc = sys.argv[1]
    fresh = "--fresh" in sys.argv

    was_cached = _cached(carc) is not None and not fresh
    results = search_code(carc, use_cache=not fresh)
    print(f"CARC {carc}   {len(results)} result(s)"
          f"{'  (from cache)' if was_cached else ''}\n")
    print(format_for_agent(carc, results))


if __name__ == "__main__":
    main()
