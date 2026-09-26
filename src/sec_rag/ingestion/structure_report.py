"""Turn one saved Azure Layout response into an explanatory text report.

This module is deliberately pure: it receives only the raw dictionary saved by
the parser and returns text. Keeping it separate from `inspection.py` lets a
reader understand file selection and safe local I/O before reading the Azure
structure interpretation used for the Stage 0 spike.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

HEADING_ROLES = {"title", "sectionHeading"}
NOISE_ROLES = {"pageHeader", "pageFooter", "pageNumber"}
MAX_DEPTH = 5


@dataclass
class Heading:
    """Keep a heading's location and two independent hierarchy signals.

    `offset` restores document reading order. `tree_level` and
    `markdown_level` instead describe nesting; retaining both lets the report
    expose disagreement in Azure's output before Stage 1.3 corrects it.
    """

    text: str
    page: int
    offset: int
    markdown_level: int | None
    tree_level: int | None


def build_structure_report(doc_name: str, raw: dict[str, Any]) -> str:
    """Render the report used to inspect heading attribution before chunking.

    The raw response remains unchanged. This implements the Stage 1.1 decision
    that `structure.txt` is a reproducible inspection aid, rather than a cache
    whose contents affect whether Azure must be called again.
    """
    paragraphs = raw.get("paragraphs") or []
    sections = raw.get("sections") or []
    content = raw.get("content") or ""
    headings = find_headings(paragraphs, sections, content)
    # Prefer Azure's explicit section tree. Use Markdown only when the tree did
    # not identify the paragraph; zero is a valid root depth, not a missing value.
    levels = [
        heading.tree_level
        if heading.tree_level is not None
        else heading.markdown_level or 1
        for heading in headings
    ]
    page_map = build_page_map(
        [
            (heading.page, level, heading.text)
            for heading, level in zip(headings, levels)
        ],
        len(raw["pages"]),
    )
    parts = [
        f"# Structure report: {doc_name}",
        role_census(paragraphs),
        sections_summary(sections, paragraphs),
        heading_table(headings),
        page_map_table(page_map),
        page_count_check(len(raw["pages"])),
    ]
    return "\n\n".join(parts) + "\n"


# Azure hierarchy extraction -------------------------------------------------
# These helpers retain both hierarchy signals so the report exposes disagreement
# for human review instead of concealing it in a guessed normalisation rule.
def find_headings(
    paragraphs: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    content: str,
) -> list[Heading]:
    """Join Azure's flat section references back to semantic paragraphs.

    `section_depths()` first calculates the level of each section. A section's
    first `/paragraphs/N` element identifies the paragraph that introduces it,
    so `tree_level_of` maps that paragraph index to the section's depth.
    Only Azure's `title` and `sectionHeading` roles become headings.
    """
    depths = section_depths(sections)

    # Azure stores sections in one flat list. References such as
    # `/paragraphs/26` connect a section to the paragraph that heads it.
    tree_level_of: dict[int, int] = {}
    for index, section in enumerate(sections):
        elements = section.get("elements") or []
        if elements and elements[0].startswith("/paragraphs/"):
            paragraph_index = int(elements[0].split("/")[-1])
            tree_level_of[paragraph_index] = depths[index]

    headings: list[Heading] = []
    for index, paragraph in enumerate(paragraphs):
        if paragraph.get("role") not in HEADING_ROLES:
            continue
        # Offset locates the heading in the complete Markdown string. It is
        # used for reading order and to find that line's Markdown `#` level.
        span = (paragraph.get("spans") or [{}])[0]
        offset = span.get("offset", 0)
        headings.append(
            Heading(
                text=" ".join(str(paragraph.get("content", "")).split()),
                page=first_page(paragraph),
                offset=offset,
                markdown_level=markdown_level(content, offset),
                tree_level=tree_level_of.get(index),
            )
        )
    return sorted(headings, key=lambda heading: heading.offset)


def section_depths(sections: list[dict[str, Any]]) -> list[int]:
    """Turn Azure's flat `/sections/N` references into nesting depths.

    If section 0 contains `/sections/1`, section 1 is its child and receives
    depth 1. This single forward pass relies on Azure listing parents before
    children, as observed in the Stage 0 filing.
    """
    depths = [0] * len(sections)
    for index, section in enumerate(sections):
        for element in section.get("elements") or []:
            if element.startswith("/sections/"):
                depths[int(element.split("/")[-1])] = depths[index] + 1
    return depths


def markdown_level(content: str, offset: int) -> int | None:
    """Count hashes on the heading line as a second hierarchy signal.

    Azure's span may begin on the heading text rather than its opening hashes,
    so inspect the complete line containing `offset`.
    """
    line_start = content.rfind("\n", 0, offset) + 1
    line_end = content.find("\n", offset)
    line = content[line_start : line_end if line_end != -1 else len(content)]
    hashes = len(line) - len(line.lstrip("#"))
    return hashes or None


def first_page(paragraph: dict[str, Any]) -> int:
    """Return where a heading starts when it spans one or more pages."""
    regions = paragraph.get("boundingRegions") or [{}]
    return regions[0].get("pageNumber", 0)


def build_page_map(
    headings: list[tuple[int, int, str]], page_count: int
) -> dict[int, list[str]]:
    """Carry each heading path forward until a sibling or ancestor replaces it.

    This applies the Stage 1.3 attribution rule for checking the future
    chunker's page ranges: a heading covers subsequent pages until another
    heading at the same or shallower level begins. The stack holds the active
    path from outermost to innermost heading.
    """
    stack: list[tuple[int, str]] = []
    page_map: dict[int, list[str]] = {}
    remaining = iter(headings)
    upcoming = next(remaining, None)
    for page in range(1, page_count + 1):
        while upcoming is not None and upcoming[0] <= page:
            _, level, text = upcoming
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            upcoming = next(remaining, None)
        page_map[page] = [text for _, text in stack][:MAX_DEPTH]
    return page_map


# Report rendering -----------------------------------------------------------
# Each renderer maps one diagnostic question in the Stage 0/1 inspection report
# to text, keeping `build_structure_report()` a readable outline of the output.
def role_census(paragraphs: list[dict[str, Any]]) -> str:
    """Report Azure roles and quantify the repeated page noise to strip later."""
    counts = Counter(paragraph.get("role") or "(body text)" for paragraph in paragraphs)
    noise = sum(counts[role] for role in NOISE_ROLES)
    lines = [f"  {role:18} {count:6}" for role, count in counts.most_common()]
    share = noise / max(len(paragraphs), 1)
    return "\n".join(
        [
            "## 1. Role census",
            *lines,
            f"  -> {noise} of {len(paragraphs)} paragraphs are header/footer/"
            f"page-number noise to strip ({share:.0%})",
        ]
    )


def sections_summary(
    sections: list[dict[str, Any]], paragraphs: list[dict[str, Any]]
) -> str:
    """Report whether the Azure section tree nests and has source spans."""
    depths = section_depths(sections)
    with_spans = sum(1 for section in sections if section.get("spans"))
    nested = sum(
        1
        for section in sections
        if any(
            element.startswith("/sections/")
            for element in section.get("elements") or []
        )
    )
    histogram = ", ".join(
        f"d{depth}={count}" for depth, count in sorted(Counter(depths).items())
    )
    lines = [
        "## 2. Sections tree",
        f"  sections: {len(sections)}   max depth: {max(depths, default=0)}   "
        f"containing sub-sections: {nested}",
        f"  sections with spans populated: {with_spans} of {len(sections)}",
        f"  depth histogram: {histogram}",
        "  first 40 sections, indented by depth (heading text, start page):",
    ]
    for index, section in enumerate(sections[:40]):
        lines.append(
            f"    {'  ' * depths[index]}[{index}] {section_label(section, paragraphs)}"
        )
    return "\n".join(lines)


def section_label(section: dict[str, Any], paragraphs: list[dict[str, Any]]) -> str:
    """Describe a section by its initial paragraph when Azure provides one."""
    elements = section.get("elements") or []
    if not elements or not elements[0].startswith("/paragraphs/"):
        return "(no leading paragraph)"
    paragraph = paragraphs[int(elements[0].split("/")[-1])]
    text = " ".join(str(paragraph.get("content", "")).split())[:70]
    return f"{text!r}  p{first_page(paragraph)}  role={paragraph.get('role') or 'body'}"


def heading_table(headings: list[Heading]) -> str:
    """Render each heading and whether its hierarchy signals agree."""
    gaps = Counter(
        heading.markdown_level - heading.tree_level
        for heading in headings
        if heading.markdown_level is not None and heading.tree_level is not None
    )
    gap_summary = (
        ", ".join(f"{gap:+d} x{count}" for gap, count in sorted(gaps.items()))
        or "none comparable"
    )
    items = [
        heading for heading in headings if heading.text.lower().startswith("item ")
    ]
    item_levels = sorted(
        Counter(heading.tree_level for heading in items).items(), key=str
    )
    lines = [
        "## 3. Headings",
        f"  headings found: {len(headings)}   with a tree level: "
        f"{sum(heading.tree_level is not None for heading in headings)}   "
        f"with a markdown level: "
        f"{sum(heading.markdown_level is not None for heading in headings)}",
        f"  markdown '#' count minus tree depth, per heading: {gap_summary}",
        f"  'Item ...' headings: {len(items)}, at (tree level, count): {item_levels}",
        "  page  tree  md   text",
    ]
    for heading in headings:
        lines.append(
            f"  {heading.page:4}  {str(heading.tree_level):>4}  "
            f"{str(heading.markdown_level):>3}   {heading.text[:80]}"
        )
    return "\n".join(lines)


def page_map_table(page_map: dict[int, list[str]]) -> str:
    """Render Azure page numbers, FinanceBench indices and active heading paths."""
    lines = [
        "## 4. Page -> heading path",
        "  azure = Azure pageNumber (1-indexed); fb = FinanceBench evidence_page_num",
        "  azure   fb   heading path",
    ]
    for page, path in page_map.items():
        shown = " > ".join(path) or "(before any heading)"
        lines.append(f"  {page:5} {page - 1:4}   {shown}")
    return "\n".join(lines)


def page_count_check(azure_page_count: int) -> str:
    """Record the page count whose PDF equality inspection already validated."""
    return "\n".join(
        [
            "## 5. Page-count check",
            f"  Azure page count: {azure_page_count}",
            "  Source-PDF equality: validated before this report was rendered",
        ]
    )
