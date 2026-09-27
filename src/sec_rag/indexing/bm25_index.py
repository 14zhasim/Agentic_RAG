"""Build, save and reload the BM25 keyword index over every chunk (Build Order 1.5).

LlamaIndex's `BM25Retriever` wraps the `bm25s` library: it splits each
chunk's text into words, drops stopwords, stems what is left, and scores a
question's words against them. Two things the wrapper does by default are
wrong for us (guide 1.5-1.6, BM25 findings 1 and 2):

- its saved index forgets our word-splitting pattern, so reloading goes
  through `bm25s` directly and re-applies the config's settings;
- it would index metadata as words, which `nodes.chunks_to_nodes` prevents.

A third fix is ours (finding 6): HTML table tags are removed from the text
BM25 counts, so tables are not scored down for their markup.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path
from typing import Any

import bm25s
import Stemmer
from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever

from ..ingestion.parse import select_documents
from .nodes import load_corpus_nodes


def build_bm25_index(config: dict[str, Any]) -> dict[str, int]:
    """Index every filing's chunks with BM25, save to indexes/bm25/, return counts.

    Always a full rebuild: it takes about a minute and is free. The new index
    is saved beside the old one and swapped in only once complete, so a
    crash never leaves half an index.
    """
    # Copied into a list[BaseNode], the type from_defaults declares; mypy
    # won't accept a list[TextNode] in its place, though every item fits.
    nodes: list[BaseNode] = list(load_corpus_nodes(config))
    for node in nodes:
        node.set_content(words_only(node.get_content()))
    settings = config["bm25"]
    # from_defaults (bm25_retriever.md lines 80-110) calls bm25s.tokenize with
    # our pattern, stopwords and stemmer, then bm25s.BM25().index(...)
    # (bm25/base.py lines 104-112). bm25s lowercases first, so the pattern
    # splits "FY2018" into "fy" and "2018", and a question's "2018" matches.
    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        token_pattern=settings["token_pattern"],
        language=settings["stopwords"],
        stemmer=Stemmer.Stemmer(settings["stemmer"]),
    )
    # k1=1.5 and b=0.75 are bm25s's defaults (bm25s README, "BM25 variants");
    # the wrapper does not expose them, and they are the design's values
    # (Build Order 1.5).
    bm25_dir = _bm25_dir(config)
    partial_dir = bm25_dir.with_name("bm25.partial")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    # persist (bm25/base.py lines 211-217) = bm25s's own save (arrays, the
    # node corpus, vocabulary) + retriever.json.
    retriever.persist(str(partial_dir))
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)
    partial_dir.rename(bm25_dir)
    return {"filings": len(select_documents(config, None)), "chunks": len(nodes)}


def load_bm25_index(
    config: dict[str, Any], similarity_top_k: int = 10
) -> BM25Retriever:
    """Reload the saved index with the config's word-splitting pattern and stemmer.

    Not `BM25Retriever.from_persist_dir`: retriever.json holds only
    DEFAULT_PERSIST_ARGS (bm25/base.py line 32), so that would tokenise every
    question with the default pattern `(?u)\\b\\w\\w+\\b`, which keeps
    "fy2018" whole and never matches "2018" (BM25 finding 1; pinned by
    test_bm25_index.py). Instead the arrays are loaded with bm25s and handed
    to a new retriever with our settings.
    """
    bm25_dir = _bm25_dir(config)
    if not bm25_dir.is_dir():
        raise ValueError("BM25 index not built yet: run sec-rag index-bm25")
    # bm25s README, "save/load": load_corpus=True brings back the node
    # records, which the wrapper turns into search results.
    bm25 = bm25s.BM25.load(str(bm25_dir), load_corpus=True)
    settings = config["bm25"]
    # existing_bm25 skips re-indexing (bm25/base.py lines 92-94). The
    # pattern and stemmer are applied to every question (lines 229-236);
    # stopwords there are bm25s's default English list, the same words the
    # "en" setting removed at build time.
    return BM25Retriever(
        existing_bm25=bm25,
        similarity_top_k=similarity_top_k,
        token_pattern=settings["token_pattern"],
        stemmer=Stemmer.Stemmer(settings["stemmer"]),
    )


def words_only(text: str) -> str:
    """Return a chunk's text with its HTML markup removed, for BM25 to count.

    Azure writes tables and figures as HTML, and BM25 would count every tag
    as a word: in 3M 2018's cash-flow table 432 of 823 words are td/tr/th,
    and tags are 12% of all words in the corpus. BM25 scores a long chunk
    down (b=0.75), so the tags push tables below prose (guide 1.5-1.6, BM25
    finding 6). Only BM25's copy changes; the chunk files, Chroma and the
    answer prompt keep the HTML, which shows the model rows and columns.

    Only the seven tags Azure's markdown output contains are removed, so a
    "<" in prose is left alone; entities such as "&amp;" become "&", so
    "PP&amp;E" doesn't add the word "amp".
    """
    return html.unescape(_TAG.sub(" ", text))


# The only tags in the 21,039 chunks (counted over every chunk file).
_TAG = re.compile(r"</?(?:table|tr|td|th|caption|figure|figcaption)\b[^>]*>")


def _bm25_dir(config: dict[str, Any]) -> Path:
    """Where the BM25 index lives: <indexes_dir>/bm25."""
    return Path(config["corpus"]["indexes_dir"]) / "bm25"
