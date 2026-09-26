"""Load and prepare each page's content for chunking.

Turns one saved Azure parse into one record per page: the page's markdown
text with page headers, footers, page numbers and page breaks blanked out,
its FinanceBench page number, its raw start position in Azure's `content`,
and the filing's company, year and type. Free and offline: it only reads the
saved JSON, which it never edits.

The rule that shapes everything here (`Systems Design Draft.md` -> "Reading
the parsed JSON"): every heading and chunk is located by its raw position in
`content`, so cleaning must never shift a position. Markers are therefore
replaced by the same number of spaces rather than cut out, and the text is
never trimmed, so position `i` in a page's text is raw position
`start_offset + i`.
"""

from __future__ import annotations

import re
from typing import Any

from .parse import json_path, read_json

# Azure writes page noise into the markdown as HTML comments, e.g.
# <!-- PageHeader="Table of Contents" -->, <!-- PageNumber="13" -->,
# <!-- PageBreak --> (docs/libraries/azure-di/markdown-output.md ->
# "PageNumber/PageHeader/PageFooter"). Non-greedy, and DOTALL so "." also
# matches a newline, in case a header's quoted text runs over two lines.
PAGE_MARKER = re.compile(r"<!--\s*Page(Header|Footer|Number|Break)\b.*?-->", re.DOTALL)


def load_pages(config: dict[str, Any], doc_name: str) -> list[dict[str, Any]]:
    """Load one filing's saved parse and prepare each page's content for chunking, in page order.

    Each page is sliced from the untouched `content` using the page's span,
    then its markers are blanked. A blank page (no span) gets empty text and
    no start position. A page with more than one span stops the load: it has
    never been seen (3M has exactly one per page), and joining spans would
    break the `start_offset + i` rule silently.
    """
    path = json_path(config, doc_name)
    if not path.is_file():
        raise ValueError(f"{doc_name}: not parsed yet")
    raw = read_json(path)
    company, year, doc_type = _split_doc_name(doc_name)
    content = raw["content"]

    pages: list[dict[str, Any]] = []
    for page in raw["pages"]:
        spans = page.get("spans") or []
        if not spans:
            start_offset: int | None = None
            text = ""
        elif len(spans) > 1:
            raise ValueError(
                f"{doc_name}: page {page['pageNumber']} has {len(spans)} spans"
            )
        else:
            start_offset = spans[0]["offset"]
            end = start_offset + spans[0]["length"]
            text = _blank_page_markers(content[start_offset:end])
        pages.append(
            {
                "doc_name": doc_name,
                "company": company,
                "year": year,
                "doc_type": doc_type,
                # Azure's pageNumber counts from 1; FinanceBench's
                # evidence_page_num counts from 0. Design draft: "never use the
                # page number printed in the footer".
                "page_index": page["pageNumber"] - 1,
                "start_offset": start_offset,
                "text": text,
            }
        )
    return pages


def _split_doc_name(doc_name: str) -> tuple[str, int, str]:
    """Split COMPANY_YEAR_TYPE into its three parts.

    Read from the right, because a company name can itself contain "_"
    (`JOHNSON_JOHNSON_2022_10K`). The filename is the source of the metadata,
    not parsing, and it is the same name the model picks from when choosing a
    filing, so chunk metadata and the model's choice always match.
    """
    parts = doc_name.rsplit("_", 2)
    if len(parts) != 3 or not (parts[1].isdigit() and len(parts[1]) == 4):
        raise ValueError(f"{doc_name}: filename must end in _<year>_<type>")
    company, year, doc_type = parts
    return company, int(year), doc_type


def _blank_page_markers(text: str) -> str:
    """Replace each page marker with spaces, so the text keeps its exact length.

    Every character of a marker becomes a space except a newline, which is
    kept, so line structure is unchanged too. The result is never trimmed.
    """
    return PAGE_MARKER.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), text)
