"""
Day 6: real retrieval.

Replaces the hardcoded string in retrieve_policy() with an actual search over a
document corpus.

Two backends behind one interface:

  embedding   sentence-transformers, all-MiniLM-L6-v2, runs locally.
              No API calls, so it costs nothing against the daily quota.
  tfidf       scikit-learn TF-IDF with cosine similarity. Used automatically
              if sentence-transformers isn't installed.

The fallback is deliberate. The retriever is an interface; the backend is a
choice. Being able to swap it without touching the agent is the point, and it
also means this runs on a machine where the torch install failed.

    python retrieval.py                      # build the index, run test queries
    python retrieval.py "prior authorization not on claim"
"""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CORPUS_DIR = Path("policies")
GENERAL_PENALTY = 0.5
INDEX_PATH = Path("policies") / ".index.pkl"


@dataclass
class Chunk:
    doc: str          # filename it came from
    heading: str      # section heading
    text: str         # the chunk body
    category: Optional[str]  # denial category, if the doc declares one

    def display(self) -> str:
        return f"[{self.doc} :: {self.heading}]\n{self.text}"


# ------------------------------------------------------------------ chunking

def chunk_document(path: Path) -> list[Chunk]:
    """Split on markdown headings. Policy docs are short and already sectioned,
    so the author's own structure is a better boundary than a fixed token count."""
    raw = path.read_text(encoding="utf-8")

    category = None
    m = re.search(r"^Category:\s*(\w+)", raw, re.MULTILINE)
    if m:
        category = m.group(1)

    title_match = re.match(r"^#\s+(.+)", raw)
    title = title_match.group(1).strip() if title_match else path.stem

    body = raw[title_match.end():] if title_match else raw

    chunks: list[Chunk] = []
    parts = re.split(r"^##\s+(.+)$", body, flags=re.MULTILINE)

    # parts[0] is the text before the first ## heading. The CARC reference docs
    # have no ## sections at all, so without this they become one long chunk and
    # length normalisation buries them under short unrelated ones.
    preamble = parts[0].strip()
    if preamble:
        paras = [p.strip() for p in re.split(r"\n\s*\n", preamble) if p.strip()]
        if len(parts) == 1 and len(paras) > 1:
            for n, para in enumerate(paras, 1):
                first = para.split(".")[0][:60]
                chunks.append(
                    Chunk(path.name, first, f"{title}\n\n{para}", category)
                )
        else:
            chunks.append(Chunk(path.name, title, f"{title}\n\n{preamble}", category))

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            chunks.append(
                Chunk(path.name, heading, f"{title} — {heading}\n\n{text}", category)
            )

    return chunks


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    if not corpus_dir.exists():
        raise SystemExit(
            f"{corpus_dir}/ not found. Run: python build_corpus.py"
        )
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_document(path))
    if not chunks:
        raise SystemExit(f"no documents found in {corpus_dir}/")
    return chunks


# ------------------------------------------------------------------ backends

class TfidfBackend:
    name = "tfidf"

    def __init__(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   sublinear_tf=True)
        self.matrix = self.vec.fit_transform(texts)

    def scores(self, query: str):
        from sklearn.metrics.pairwise import cosine_similarity
        qv = self.vec.transform([query])
        return cosine_similarity(qv, self.matrix)[0]


class EmbeddingBackend:
    name = "embedding (all-MiniLM-L6-v2)"

    def __init__(self, texts: list[str]):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.matrix = self.model.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False)

    def scores(self, query: str):
        qv = self.model.encode([query], normalize_embeddings=True,
                               show_progress_bar=False)
        return (self.matrix @ qv[0])


def make_backend(texts: list[str], prefer: str = "embedding"):
    if prefer == "embedding":
        try:
            return EmbeddingBackend(texts)
        except ImportError:
            print("  sentence-transformers not installed, using tfidf")
        except Exception as exc:
            print(f"  embedding backend failed ({type(exc).__name__}), using tfidf")
    return TfidfBackend(texts)


# ------------------------------------------------------------------ retriever

class PolicyRetriever:
    def __init__(self, corpus_dir: Path = CORPUS_DIR, prefer: str = "embedding"):
        self.chunks = load_chunks(corpus_dir)
        self.backend = make_backend([c.text for c in self.chunks], prefer)

    def search(self, query: str, k: int = 3,
               category: Optional[str] = None,
               min_score: float = 0.05,
               max_per_doc: int = 99) -> list[tuple[Chunk, float]]:
        """Return the top k chunks. If a category is given, chunks tagged with a
        different category are dropped — a metadata filter over the vector
        search, so a timely-filing query can't surface a coverage policy."""
        scores = self.backend.scores(query)

        # Chunks tagged "any" are general appeal guidance. When the caller has
        # named a category, a short general passage should not outrank a
        # document written about the thing being asked, so discount it.
        #
        # When no category is given the query IS general, and the penalty was
        # halving the only document that could answer it. The retrieval eval
        # caught this: three queries about appeal format, all missing the
        # appeal format policy. The penalty only makes sense relative to
        # topical competition that exists.
        penalise_general = category is not None
        adjusted = [
            (c, s * (GENERAL_PENALTY
                     if (penalise_general and c.category == "any") else 1.0))
            for c, s in zip(self.chunks, scores)
        ]
        ranked = sorted(adjusted, key=lambda p: -p[1])

        out: list[tuple[Chunk, float]] = []
        seen_docs: dict[str, int] = {}
        for chunk, score in ranked:
            if score < min_score:
                continue
            # Without this the top k comes back as k chunks of one document and
            # the rest of the corpus is never seen. That is how a policy stating
            # the appeal route was withdrawn got missed on a claim about that
            # exact appeal route.
            if seen_docs.get(chunk.doc, 0) >= max_per_doc:
                continue
            # A chunk tagged "any" is general guidance and always eligible.
            # A chunk with no tag at all is a corpus bug: every document should
            # declare one, and an untagged chunk is what let the non-covered
            # policy surface on a medical necessity query.
            if category and chunk.category not in (None, "any", category):
                continue
            if category and chunk.category is None:
                continue
            out.append((chunk, float(score)))
            seen_docs[chunk.doc] = seen_docs.get(chunk.doc, 0) + 1
            if len(out) >= k:
                break
        return out

    def retrieve_for_agent(self, query: str, category: Optional[str] = None,
                           k: int = 3,
                           extra_queries: Optional[list[str]] = None,
                           max_per_doc: int = 2) -> str:
        """What the tool hands back to the model. Every passage is labelled with
        the document it came from, so the model can't present retrieved text as
        its own knowledge and a human can check it.

        The claim text alone describes what the claim IS. The agent needs to
        know what it can DO about it, and documents answering that second
        question share little vocabulary with the first. So the caller can pass
        extra_queries and the results are merged, each chunk keeping its best
        score across queries."""
        queries = [query] + (extra_queries or [])

        best: dict[int, tuple[Chunk, float]] = {}
        for q in queries:
            for chunk, score in self.search(q, k=k * 2, category=category):
                key = id(chunk)
                if key not in best or score > best[key][1]:
                    best[key] = (chunk, score)

        ranked = sorted(best.values(), key=lambda p: -p[1])

        hits: list[tuple[Chunk, float]] = []
        per_doc: dict[str, int] = {}
        for chunk, score in ranked:
            if per_doc.get(chunk.doc, 0) >= max_per_doc:
                continue
            hits.append((chunk, score))
            per_doc[chunk.doc] = per_doc.get(chunk.doc, 0) + 1
            if len(hits) >= k:
                break
        if not hits:
            return (
                "No policy passages matched this query. Do not assert policy "
                "language that was not retrieved."
            )

        blocks = []
        for chunk, score in hits:
            blocks.append(f"{chunk.display()}\n(relevance {score:.2f})")
        return (
            "Retrieved policy passages. Quote only from these; anything not "
            "here was not retrieved.\n\n" + "\n\n---\n\n".join(blocks)
        )


_RETRIEVER: Optional[PolicyRetriever] = None


def get_retriever() -> PolicyRetriever:
    """Built once per process. Indexing every call would dominate runtime."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = PolicyRetriever()
    return _RETRIEVER


# ------------------------------------------------------------------ cli

TEST_QUERIES = [
    ("authorization was obtained but not included on the claim", "authorization_missing"),
    ("service excluded by the benefit plan, documentation on file", "noncovered_charge"),
    ("claim filed after the deadline", "timely_filing"),
    ("clinical records supporting medical necessity", "medical_necessity"),
]


def main() -> None:
    print("building index...")
    r = get_retriever()
    print(f"  backend:  {r.backend.name}")
    print(f"  chunks:   {len(r.chunks)} from "
          f"{len(set(c.doc for c in r.chunks))} documents\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"query: {query}\n")
        print(r.retrieve_for_agent(query))
        return

    for query, category in TEST_QUERIES:
        print("=" * 70)
        print(f"query:    {query}")
        print(f"category: {category}")
        for chunk, score in r.search(query, k=2, category=category):
            print(f"  {score:.3f}  {chunk.doc} :: {chunk.heading}")
        print()


if __name__ == "__main__":
    main()
