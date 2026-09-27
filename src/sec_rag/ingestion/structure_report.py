"""Turn one saved Azure parse into a readable structure report.

The report is for eyeballing only: nothing downstream reads it. It answers,
for one filing, "did Azure find the headings and nest them sensibly?", so a
person can judge the parse before the rest of the corpus is paid for.

This module touches no files — saved JSON in, report text out — so it can be
tested with a small made-up dictionary. The heading list itself comes from
`headings.extract_headings`; this module only summarises and formats it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .headings import Heading, extract_headings


def build_structure_report(doc_name: str, raw: dict[str, Any]) -> str:
    """Build the three-section structure report for one saved parse.

    Each section answers one question asked before paying for the rest of
    the corpus: what headings did Azure find, what does their nesting do to
    each page, and did Azure return any figures?
    """
    page_count = len(raw.get("pages") or [])
    headings = extract_headings(raw)
    parts = [
        f"# Structure report: {doc_name}",
        heading_table(headings),
        page_map_table(build_page_map(headings, page_count)),
        figure_count(raw),
    ]
    return "\n\n".join(parts) + "\n"


def heading_table(headings: list[Heading]) -> str:
    """Section 1: one line per heading, plus where the 'Item ...' headings sit.

    On a well-structured 10-K all Items share one level; on 3M 2018 they sit
    at four, which is what Stage 1.3's heading-fix pass corrects.
    """
    outside_tree = sum(1 for heading in headings if heading.level is None)
    items = [
        heading for heading in headings if heading.text.lower().startswith("item ")
    ]
    item_levels = Counter(heading.level for heading in items)
    item_summary = ", ".join(
        f"level {level}: {count}"
        for level, count in sorted(item_levels.items(), key=lambda pair: str(pair[0]))
    )
    lines = [
        "## 1. Headings",
        f"  headings: {len(headings)}   outside the section tree (no level): "
        f"{outside_tree}",
        f"  'Item ...' headings: {len(items)} ({item_summary or 'none'})",
        "    offset  page  level  text",
    ]
    for heading in headings:
        lines.append(
            f"  {heading.offset:8}  {heading.page:4}  {str(heading.level):>5}  "
            f"{heading.text[:80]}"
        )
    return "\n".join(lines)


def build_page_map(headings: list[Heading], page_count: int) -> dict[int, list[str]]:
    """Work out each page's heading path by carrying headings across pages.

    A heading stays on the path until a heading at the same or a shallower
    level starts: that is the attribution rule from `Build Order.md` §1.3,
    applied per page for eyeballing. The path is kept at full depth, so the
    report shows exactly what Azure returned. Headings with no level are left
    out, since there is no way to place them on the path.
    """
    # (page, level, text) for every heading that has a level, in reading order.
    placed = [
        (heading.page, heading.level, heading.text)
        for heading in headings
        if heading.level is not None
    ]
    path: list[tuple[int, str]] = []  # (level, text), outermost first
    page_map: dict[int, list[str]] = {}
    next_index = 0
    for page in range(1, page_count + 1):
        while next_index < len(placed) and placed[next_index][0] <= page:
            _, level, text = placed[next_index]
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, text))
            next_index += 1
        page_map[page] = [text for _, text in path]
    return page_map


def page_map_table(page_map: dict[int, list[str]]) -> str:
    """Section 2: each page's heading path, with Azure's and FinanceBench's page numbers."""
    lines = [
        "## 2. Page -> heading path",
        "  azure = Azure pageNumber (counts from 1); fb = FinanceBench "
        "evidence_page_num (counts from 0)",
        "  azure   fb   heading path",
    ]
    for page, path in page_map.items():
        shown = " > ".join(path) or "(before any heading)"
        lines.append(f"  {page:5} {page - 1:4}   {shown}")
    return "\n".join(lines)


def figure_count(raw: dict[str, Any]) -> str:
    """Section 3: did Azure return any figures?

    Figure text stays in the page text and is chunked like prose (design
    draft, "Figure text kept as ordinary page text"), so this is a check, not
    a step. Azure wraps figures in `<figure>` tags in `content` and may also
    list them in a `figures` field, which is absent when there are none
    (`docs/libraries/azure-di/markdown-output.md` -> "Figure").
    """
    content = raw.get("content") or ""
    tags = content.count("<figure>")
    listed = len(raw.get("figures") or [])
    return "\n".join(
        [
            "## 3. Figures",
            f"  <figure> tags in content: {tags}",
            f"  entries in Azure's figures list: {listed}",
        ]
    )
