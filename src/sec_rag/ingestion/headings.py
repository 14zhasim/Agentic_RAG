"""Read the heading list out of one saved Azure parse.

Azure has already worked out the structure: every paragraph arrives with a
role, and the flat `sections` list records which section contains which. This
module only reads those labels into one row per heading — raw offset, page,
level and text — for two consumers: the inspection report now, and Stage
1.3's heading-fix pass and chunker later (`Systems Design Draft.md` -> "The
heading list: `extract_headings`").

Levels are Azure's raw levels, exactly as its section tree gives them, at full
depth. Correcting them, and the five-level cap, happen in Stage 1.3 after the
heading-fix pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HEADING_ROLES = {"title", "sectionHeading"}


@dataclass(frozen=True)
class Heading:
    """One heading in a filing, located by its raw position in Azure's content.

    `offset` is the heading's permanent ID: later stages rename and re-level
    headings but never change it, and chunks are matched to headings by it.
    """

    offset: int  # raw position in Azure's content where the heading's line starts
    page: int  # Azure pageNumber the heading starts on (counts from 1)
    level: int | None  # depth in Azure's section tree; None if no section opens with it
    text: str  # the heading's words, whitespace tidied


def extract_headings(raw: dict[str, Any]) -> list[Heading]:
    """List every heading in a saved parse with its raw position, page, level and text, in reading order.

    Azure splits what we need across two lists: `paragraphs` know each
    heading's text, page and position but not its depth; `sections` know the
    nesting but refer to paragraphs only by number. Three steps join them:

    1. `section_depths` gives each section its nesting depth.
    2. Each section's depth goes to the paragraph that opens it.
    3. Paragraphs with a heading role become `Heading` rows, sorted by
       offset. A heading that opens no section gets `level=None`.
    """
    paragraphs = raw.get("paragraphs") or []
    sections = raw.get("sections") or []
    depths = section_depths(sections)

    # Step 2: a section's first element, if it's a paragraph, is its heading,
    # so that paragraph takes the section's depth. `split("/")[-1]` turns
    # "/paragraphs/4" into 4.
    level_of_paragraph: dict[int, int] = {}
    for section_index, section in enumerate(sections):
        elements = section.get("elements") or []
        if elements and elements[0].startswith("/paragraphs/"):
            paragraph_index = int(elements[0].split("/")[-1])
            level_of_paragraph[paragraph_index] = depths[section_index]

    # Step 3: keep the heading paragraphs.
    headings: list[Heading] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        if paragraph.get("role") not in HEADING_ROLES:
            continue
        # The span's offset points at the start of the heading's markdown line,
        # its "#"s included (checked on 3M: content[54616:] is "## PART II").
        offset = paragraph["spans"][0]["offset"]
        headings.append(
            Heading(
                offset=offset,
                page=paragraph["boundingRegions"][0]["pageNumber"],
                # .get() returns None for a heading outside the tree. Depth 0 is
                # a real level, so later code must test `is not None`, not truth.
                level=level_of_paragraph.get(paragraph_index),
                text=" ".join(str(paragraph.get("content", "")).split()),
            )
        )
    return sorted(headings, key=lambda heading: heading.offset)


def section_depths(sections: list[dict[str, Any]]) -> list[int]:
    """Turn Azure's flat `/sections/N` references into a nesting depth per section.

    If section 0 contains `/sections/1`, section 1 is its child and gets depth
    1. This single forward pass relies on Azure listing a parent before its
    children, which held on 3M 2018.
    """
    depths = [0] * len(sections)
    for index, section in enumerate(sections):
        for element in section.get("elements") or []:
            if element.startswith("/sections/"):
                depths[int(element.split("/")[-1])] = depths[index] + 1
    return depths
