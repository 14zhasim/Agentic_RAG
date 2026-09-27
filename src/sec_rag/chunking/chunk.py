"""Cut one parsed filing into page-bounded chunks.

The rules are `Systems Design Draft.md` -> Ingest files -> Chunk, rules 1-6,
applied one page at a time:

1. no chunk crosses a page;
2. every table and figure is its own chunk, never split (a figure with no
   text is skipped);
3. the page's remaining prose is joined and cut at Azure's heading positions,
   keeping a cut only if both pieces clear the floor; anything still over the
   ceiling is halved at the sentence start nearest its middle, recursively;
4. a chunk starting at a page break or a halving cut begins with the last
   sentence before the cut (text only: its page and offsets stay its own);
5. each chunk records its page, filing, raw offsets and an empty heading path.

This module touches no files: the saved Azure JSON and `load_pages`'s pages
go in, chunk records come out, so tests can feed it a hand-made page. The
files are handled by `chunk_files.py`. The step-by-step explanation, with 3M
2018 page index 12 worked through, is in `docs sys design/Exp 1/
Implementation Guide 1.3 Chunk.md` -> "How `chunk.py` fits together".

One idea runs through it, the same one `load_pages` uses for page markers:
blank things out, don't cut them out. Tables and figures are removed from a
page's prose by replacing their characters with spaces, so position `i` in
the prose is still raw position `page_start + i`, and every cut is a raw
offset into Azure's `content`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llama_index.core.utils import get_tokenizer, globals_helper

from ..ingestion.headings import extract_headings

# Azure wraps a figure's text in <figure>...</figure>
# (docs/libraries/azure-di/markdown-output.md -> "Figure"). A caption sits
# inside, as <figcaption>, so it survives this pattern and counts as text.
FIGURE_TAG = re.compile(r"</?figure>")


@dataclass(frozen=True)
class Element:
    """One table or figure: its raw span in Azure's content, and which it is."""

    start: int
    end: int  # exclusive
    kind: str  # "table" or "figure"


@dataclass(frozen=True)
class Piece:
    """A stretch of a page's prose, by raw offsets, and why it starts where it does.

    `starts_at` is "page" (the first prose on the page), "heading" (a heading
    cut, or a heading as the page's first visible text) or "halving" (the
    second half of a halving cut). The overlap rule reads only this label.
    """

    start: int
    end: int  # exclusive
    starts_at: str


def chunk_filing(
    raw: dict[str, Any],
    pages: list[dict[str, Any]],
    chunking: dict[str, int],
) -> list[dict[str, Any]]:
    """Cut one filing into chunk records, page by page, in reading order.

    `raw` is the saved Azure JSON, `pages` is `load_pages`'s output for the
    same filing, and `chunking` is the `[chunking]` config table. Pages are
    handled one at a time, which is how rule 1 (no chunk crosses a page) is
    kept. The only thing passed from one page to the next is the sentence to
    carry across the page break (rule 4).
    """
    tables_and_figures = _table_and_figure_spans(raw)
    heading_positions = [heading.offset for heading in extract_headings(raw)]

    records: list[dict[str, Any]] = []
    carry = ""  # the first page has nothing to carry
    for page in pages:
        page_records, carry = _chunk_page(
            page, tables_and_figures, heading_positions, carry, chunking
        )
        records.extend(page_records)
    return records


def _table_and_figure_spans(raw: dict[str, Any]) -> list[Element]:
    """List every table and figure as an Element, sorted by start.

    Each has exactly one span (checked on all 64 filings: 6,441 tables, 266
    figures), and a table's span covers exactly its <table>...</table> HTML.
    Whatever Azure lists as a figure is included, even the two NIKE cover
    "figures" whose text has no <figure> tag.
    """
    elements: list[Element] = []
    for kind, key in (("table", "tables"), ("figure", "figures")):
        for item in raw.get(key) or []:
            span = item["spans"][0]
            elements.append(
                Element(span["offset"], span["offset"] + span["length"], kind)
            )
    return sorted(elements, key=lambda element: element.start)


def _chunk_page(
    page: dict[str, Any],
    tables_and_figures: list[Element],
    heading_positions: list[int],
    carry: str,
    chunking: dict[str, int],
) -> tuple[list[dict[str, Any]], str]:
    """Cut one page into chunk records, and return the sentence to carry onto the next page."""
    if page["start_offset"] is None:  # a blank page: no span, no text
        return [], ""  # a sentence can't continue across it
    page_start = page["start_offset"]
    page_text = page["text"]
    page_end = page_start + len(page_text)

    # Step 1: what's on this page. A table or figure belongs to the page its
    # start falls in; none crosses a page, so the start is enough.
    on_page = [
        element
        for element in tables_and_figures
        if page_start <= element.start < page_end
    ]

    # Step 2 (rule 2): each table and figure becomes a chunk as it is, then
    # its characters are blanked out of the prose so nothing shifts.
    # (start, end, kind, own text, carried text), one per chunk on the page
    pieces: list[tuple[int, int, str, str, str]] = []
    prose = page_text
    for element in on_page:
        element_text = page_text[element.start - page_start : element.end - page_start]
        if element.kind == "figure" and not FIGURE_TAG.sub("", element_text).strip():
            pass  # an empty figure (likely a logo): nothing to search
        else:
            pieces.append((element.start, element.end, element.kind, element_text, ""))
        prose = _blank(prose, element.start - page_start, element.end - page_start)

    # Step 3 (rule 3): cut the joined prose at headings, then halve anything
    # over the ceiling. A heading inside a table or figure is not a cut
    # point: cuts happen only in prose.
    cut_points = [
        position
        for position in heading_positions
        if page_start <= position < page_end
        and not any(element.start <= position < element.end for element in on_page)
    ]
    prose_pieces: list[Piece] = []
    for piece in _cut_at_headings(
        prose, page_start, cut_points, chunking["floor_tokens"]
    ):
        prose_pieces.extend(
            _halve_until_fits(prose, page_start, piece, chunking["ceiling_tokens"])
        )
    # A page of only tables leaves prose that's all blanks: no prose chunk.
    prose_pieces = [
        piece for piece in prose_pieces if tidy(_text_of(prose, page_start, piece))
    ]

    # Step 4 (rule 4): overlap. "page" and "halving" pieces carry the
    # sentence before the cut; "heading" pieces carry nothing, since a new
    # section doesn't need the previous one's last sentence. The carried
    # sentence goes into the chunk's text only; its offsets stay its own.
    cap = chunking["overlap_cap_tokens"]
    for index, piece in enumerate(prose_pieces):
        carried = ""
        if piece.starts_at == "page":
            carried = carry
        elif piece.starts_at == "halving":
            previous = prose_pieces[index - 1]
            carried = _last_sentence(_text_of(prose, page_start, previous), cap)
        own_text = _text_of(prose, page_start, piece)
        pieces.append((piece.start, piece.end, "prose", own_text, carried))
    if prose_pieces:
        next_carry = _last_sentence(_text_of(prose, page_start, prose_pieces[-1]), cap)
    else:
        next_carry = ""  # no prose on this page: nothing continues past it

    # Step 5 (rule 5): records in reading order, numbered c0, c1, ... on the page.
    pieces.sort(key=lambda item: item[0])
    records = []
    for number, (start, end, kind, own_text, carried) in enumerate(pieces):
        records.append(_record(page, number, start, end, kind, own_text, carried))
    return records, next_carry


def _cut_at_headings(
    prose: str, page_start: int, cut_points: list[int], floor: int
) -> list[Piece]:
    """Split the page's prose at headings, keeping only cuts that leave both sides over the floor.

    Rule 3: "accept a heading cut only if the piece before it AND the
    remainder both clear ~250 tokens"; headings are tried left to right, and
    each is measured from the last cut kept. A rejected heading stays inside
    its piece; Stage 3 merges the heading paths.
    """
    page_end = page_start + len(prose)
    first_visible = page_start + (len(prose) - len(prose.lstrip()))

    # A heading as the page's first visible text counts as a heading start,
    # so nothing is carried onto it (3M page index 12 starts "## PART II").
    label = "heading" if first_visible in cut_points else "page"

    pieces: list[Piece] = []
    piece_start = page_start
    for position in sorted(cut_points):
        if position <= first_visible:
            continue  # nothing before it to make a piece from
        tokens_before = count_tokens(
            prose[piece_start - page_start : position - page_start]
        )
        tokens_after = count_tokens(prose[position - page_start :])
        if tokens_before >= floor and tokens_after >= floor:
            pieces.append(Piece(piece_start, position, label))
            piece_start = position
            label = "heading"
    pieces.append(Piece(piece_start, page_end, label))
    return pieces


def _halve_until_fits(
    prose: str, page_start: int, piece: Piece, ceiling: int
) -> list[Piece]:
    """Halve a piece at the sentence start nearest its middle until every part fits the ceiling.

    Rule 3: "halved at the nearest sentence boundary to its midpoint,
    recursively". The first half keeps the piece's label; the second half is
    labelled "halving", so it gets overlap.
    """
    if count_tokens(_text_of(prose, page_start, piece)) <= ceiling:
        return [piece]
    cut = _halving_point(prose, page_start, piece)
    if cut is None:
        return [piece]  # nowhere to cut (one enormous word): leave it whole
    first_half = Piece(piece.start, cut, piece.starts_at)
    second_half = Piece(cut, piece.end, "halving")
    return _halve_until_fits(prose, page_start, first_half, ceiling) + (
        _halve_until_fits(prose, page_start, second_half, ceiling)
    )


def _halving_point(prose: str, page_start: int, piece: Piece) -> int | None:
    """Return the raw position inside the piece that splits its tokens most evenly.

    Candidates are sentence starts; if the piece has none (a long list with
    no full stops), line starts; if none, word starts. "Nearest the middle"
    is measured in tokens, not characters, because a piece can hold a blank
    gap where a table was: many characters, no tokens.

    Sentence starts come from LlamaIndex's sentence finder,
    `globals_helper.punkt_tokenizer.span_tokenize(text)`, which returns each
    sentence's (start, end) character positions. It is NLTK's Punkt created
    untrained (llama_index/core/utils.py line 115), so it may end a sentence
    at "Inc." or "See Note 12."; the cost is a cut a few words off.
    """
    text = _text_of(prose, page_start, piece)
    spans = list(globals_helper.punkt_tokenizer.span_tokenize(text))
    # Every sentence start except the first: cutting there leaves text on both sides.
    candidates = [start for start, _ in spans[1:]]
    # Blanked markers can leave spaces after a newline, so both patterns skip
    # any run of blanks and land on the next visible character.
    if not candidates:
        candidates = [match.end() for match in re.finditer(r"\n\s*(?=\S)", text)]
    if not candidates:
        candidates = [match.end() for match in re.finditer(r"[ \t]+(?=\S)", text)]
    candidates = [position for position in candidates if 0 < position < len(text)]
    if not candidates:
        return None

    # Tokens before a candidate only grow as it moves right, so binary-search
    # for the first candidate with at least half the tokens before it; the
    # most even split is that one or the one just before it.
    half = count_tokens(text) / 2
    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if count_tokens(text[: candidates[middle]]) < half:
            low = middle + 1
        else:
            high = middle
    best = candidates[low]
    if low > 0:
        previous = candidates[low - 1]
        if _imbalance(text, previous) < _imbalance(text, best):
            best = previous
    return piece.start + best


def _imbalance(text: str, position: int) -> int:
    """How uneven a cut at `position` is: the token difference between the two sides."""
    return abs(count_tokens(text[:position]) - count_tokens(text[position:]))


def _last_sentence(text: str, cap: int) -> str:
    """Return the last sentence of `text`, cut down to at most `cap` tokens, for overlap.

    Rule 4: the carried sentence may be complete or broken off (a page that
    ends mid-sentence). If it's over the cap, words are dropped from its
    FRONT, so what's kept is the part right before the cut, which is what
    the next chunk continues from.
    """
    spans = list(globals_helper.punkt_tokenizer.span_tokenize(text))
    if not spans:
        return ""
    last_start = spans[-1][0]
    words = tidy(text[last_start:]).split()
    while len(words) > 1 and count_tokens(" ".join(words)) > cap:
        words.pop(0)
    return " ".join(words)


def _text_of(prose: str, page_start: int, piece: Piece) -> str:
    """The piece's characters from the page's prose (prose starts at raw position page_start)."""
    return prose[piece.start - page_start : piece.end - page_start]


def _record(
    page: dict[str, Any],
    number: int,
    start: int,
    end: int,
    kind: str,
    own_text: str,
    carried_text: str,
) -> dict[str, Any]:
    """Build one chunk record (Implementation Guide 1.3 -> "The chunk file").

    The offsets are trimmed to the first and last visible character of the
    chunk's own text, so they point at real content rather than at blanks.
    `text` is the carried sentence, one space, then the chunk's own tidied
    text: the space means a sentence broken by a page break reads as one
    sentence again ("...losses on plan assets, and..."). The carried text has
    no offsets of its own, so page metrics and Stage 3's heading attribution
    see only the chunk's own piece.
    """
    leading = len(own_text) - len(own_text.lstrip())
    trailing = len(own_text) - len(own_text.rstrip())
    text = tidy(own_text)
    if carried_text:
        text = f"{carried_text} {text}"
    return {
        "chunk_id": f"{page['doc_name']}:p{page['page_index']}:c{number}",
        "doc_name": page["doc_name"],
        "company": page["company"],
        "year": page["year"],
        "doc_type": page["doc_type"],
        "page_index": page["page_index"],
        "kind": kind,
        "start_offset": start + leading,
        "end_offset": end - trailing,
        "carried_text": carried_text,
        "text": text,
        "token_count": count_tokens(text),
        # Filled in place at Stage 3.1 from the offsets above.
        "heading_path": [],
    }


def _blank(text: str, start: int, end: int) -> str:
    """Replace text[start:end] with spaces, keeping newlines, so the length is unchanged."""
    blanked = re.sub(r"[^\n]", " ", text[start:end])
    return text[:start] + blanked + text[end:]


def count_tokens(text: str) -> int:
    """Count the tokens in the tidied text with LlamaIndex's bundled cl100k counter.

    `get_tokenizer()` (llama_index/core/utils.py line 153) builds tiktoken's
    encoder once, from the copy bundled in the package's `_static` folder, and
    caches it, so this makes no network call. Counting the tidied text means
    blank gaps left by page markers and tables don't count as tokens.
    """
    return len(get_tokenizer()(tidy(text)))


def tidy(text: str) -> str:
    """Squash the blanks left by page markers and removed tables into normal spacing.

    Runs of spaces or tabs become one space, spaces at a line's start or end
    are removed (a blanked marker often sits just before text on the same
    line), three or more newlines become two, and the ends are stripped. Only
    used on text leaving the chunker: positions are always worked out on the
    untidied text.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
