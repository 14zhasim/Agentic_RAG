"""Stage 0.1 spike: is Azure Document Intelligence's structure good enough for Exp2?

Why this exists
    Exp2 attaches a heading path (e.g. "Item 7. MD&A > Results of Operations")
    to every page-bounded chunk. That path has to come from somewhere. Azure's
    `prebuilt-layout` model claims to return a nested tree of document sections;
    if the tree is good on real 10-Ks we use it, and if it comes back flat or
    wrong we fall back to PageIndex (see `Build Order.md`, section 0.1). This
    script answers that question on two real filings before Stage 1 commits to
    either.

How it works: two separate steps
    parse    PAID. Sends each PDF to `prebuilt-layout` with markdown output and
             caches the raw JSON at data/financebench/parsed/<doc_name>.json.
             A doc whose JSON already exists is skipped, so nothing is ever
             paid for twice. This cache is the same one Stage 1 will read.
    analyse  FREE, offline, no credentials. Reads the cached JSON and writes a
             readable report to data/financebench/parsed/<doc_name>.structure.txt.
             Re-run it as often as you like.

    Splitting them means the analysis can be changed and re-run without paying
    Azure again.

What Azure's JSON looks like (the parts used here)
    content     the whole document as one markdown string
    pages       one entry per page; `pageNumber` is 1-indexed
    paragraphs  every block of text. Each has
                  - `content`          its text
                  - `role`             absent for body text, otherwise one of
                                       title, sectionHeading, pageHeader,
                                       pageFooter, pageNumber, footnote,
                                       formulaBlock
                  - `spans`            [{offset, length}] pointing into `content`
                  - `boundingRegions`  [{pageNumber, polygon}] -- where it sits
    sections    the tree. Each has `elements`, a list of JSON-pointer strings
                such as "/paragraphs/12" or "/sections/3". A section listing
                another section contains it, which is what makes it a tree.

    Confirmed in docs/libraries/azure-di/layout-model.md (the "Paragraph roles"
    and "Sections" parts), live at
    https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout
    The two extra roles (footnote, formulaBlock) are not in that page's table;
    they come from the installed SDK's `ParagraphRole` enum.

Usage (credentials come from the shell, never from this file)
    set -a; source .env; set +a
    uv run python scripts/spike_azure_structure.py parse 3M_2018_10K AMD_2022_10K
    uv run python scripts/spike_azure_structure.py analyse 3M_2018_10K AMD_2022_10K

The report answers the spike's questions
    1. role census          how much header/footer/page-number noise to strip
    2. sections tree        do sections nest, and are their spans populated?
    3. heading list         is "Item 7" at the top? Do the two level sources agree?
    4. page -> heading map  the artefact to check against the PDF by eye
    5. sanity checks        page counts, and the 1-indexed -> 0-indexed conversion
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PDF_DIR = Path("benchmarks/financebench/pdfs")
PARSED_DIR = Path("data/financebench/parsed")

# Roles Azure gives to text that repeats on every page (running titles, "Page 47",
# the company name in the margin). The design strips these before chunking
# (`Systems Design Draft.md`, parser decisions) because they add the same words
# to every chunk and so make every page look alike to both BM25 and embeddings.
NOISE_ROLES = {"pageHeader", "pageFooter", "pageNumber"}

# Roles that mark the start of a heading. `title` is normally the document's
# top heading; `sectionHeading` is every heading below it.
HEADING_ROLES = {"title", "sectionHeading"}

# The design caps a heading path at five levels; deeper headings in a 10-K are
# almost always table captions or list labels rather than real structure.
MAX_DEPTH = 5


def main() -> None:
    """Parse or analyse each named filing in turn.

    Only argument handling and delegation happen here; the work is in `parse`
    and `analyse`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["parse", "analyse"])
    parser.add_argument("doc_names", nargs="+", help="e.g. 3M_2018_10K")
    args = parser.parse_args()

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    for doc_name in args.doc_names:
        if args.step == "parse":
            parse(doc_name)
        else:
            report = analyse(doc_name)
            out = PARSED_DIR / f"{doc_name}.structure.txt"
            out.write_text(report)
            # The page map (section 4) runs to hundreds of lines, so the
            # terminal gets sections 1-3 and the file holds everything.
            print(report.split("\n\n## 4.")[0])
            print(f"\nFull report, including the page map: {out}\n")


# --- step 1: parse (paid) -------------------------------------------------


def parse(doc_name: str) -> None:
    """Send one PDF to Azure `prebuilt-layout` and cache the raw JSON result.

    How it works
        1. If the JSON is already cached, stop -- this is what guarantees each
           filing is paid for once. A full re-parse of the corpus is ~$108.
        2. Read the credentials from the environment, only now, so that the
           free `analyse` step never needs them.
        3. Upload the PDF bytes and ask for markdown output. The call is a
           long-running operation: Azure returns a poller immediately and
           `.result()` waits until the analysis finishes (a minute or two for
           a 150-page filing).
        4. Save the whole result, untouched. Raw JSON is the source of truth in
           the design -- any later decision (stripping, chunking, headings) is
           made from this file, never by calling Azure again.

    The call signature was checked against the installed
    `azure-ai-documentintelligence==1.0.2`, not copied from an example.
    """
    out = PARSED_DIR / f"{doc_name}.json"
    if out.exists():
        print(f"{doc_name}: cached at {out}, skipping (no charge)")
        return

    pdf = PDF_DIR / f"{doc_name}.pdf"
    if not pdf.exists():
        sys.exit(f"{doc_name}: no PDF at {pdf}")

    # Imported here rather than at the top of the file, so that `analyse`
    # works on a machine with no Azure credentials and no network.
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentContentFormat
    from azure.core.credentials import AzureKeyCredential

    # Values are read and passed straight to the client; they are never printed
    # or written anywhere. `set -a; source .env; set +a` puts them in the shell.
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not key or not endpoint:
        sys.exit(
            "AZURE_DOCUMENT_INTELLIGENCE_KEY / _ENDPOINT not set. "
            "Run: set -a; source .env; set +a"
        )

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    print(f"{doc_name}: sending {pdf} to prebuilt-layout (PAID)...")
    with pdf.open("rb") as f:
        # MARKDOWN makes `content` markdown (headings as '#', tables as markdown
        # tables) instead of plain text. Everything else in the result is the
        # same either way; the spans simply point into whichever string it is.
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
    result = poller.result()

    # Write under a temporary name, then rename. A rename is all-or-nothing, so
    # if the script dies half-way through writing, no half-file sits at `out`
    # for step 1 to mistake for a finished parse and skip forever.
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(result.as_dict()))
    tmp.rename(out)
    print(f"{doc_name}: {len(result.pages or [])} pages parsed, cached at {out}")


# --- step 2: analyse (free) -----------------------------------------------


@dataclass
class Heading:
    """One heading paragraph, with its level measured two independent ways.

    The two levels are kept separately on purpose: whether they agree is one
    of the things the spike is measuring.
    """

    paragraph_index: int  # position in Azure's `paragraphs` list
    text: str
    page: int  # Azure's 1-indexed page number where the heading starts
    offset: int  # character position in `content`; sorting on it gives reading order
    markdown_level: int | None  # number of '#' on the heading's markdown line
    tree_level: int | None  # depth of the heading's section in Azure's `sections` tree


def analyse(doc_name: str) -> str:
    """Build the five-part structure report for one cached filing.

    How it works
        1. Load the cached JSON (no Azure call).
        2. Collect every heading paragraph with its start page and both levels.
        3. Pick one level per heading -- the tree level if Azure gave one,
           otherwise the markdown level, otherwise treat it as top level.
        4. Turn the headings into a page -> heading path map.
        5. Render the report sections as text.
    """
    path = PARSED_DIR / f"{doc_name}.json"
    if not path.exists():
        sys.exit(f"{doc_name}: not parsed yet. Run the parse step first.")
    doc = json.loads(path.read_text())

    paragraphs = doc.get("paragraphs", [])
    sections = doc.get("sections", [])
    content = doc.get("content", "")
    n_pages = len(doc.get("pages", []))

    headings = find_headings(paragraphs, sections, content)

    # Preference order for a heading's level: the tree is Azure's own claim
    # about nesting, so it wins; markdown '#' is the fallback; a heading with
    # neither is treated as top level rather than dropped.
    levels = [h.tree_level or h.markdown_level or 1 for h in headings]
    page_map = build_page_map(
        [(h.page, level, h.text) for h, level in zip(headings, levels)], n_pages
    )

    parts = [
        f"# Structure report: {doc_name}",
        role_census(paragraphs),
        sections_summary(sections, paragraphs),
        heading_table(headings),
        page_map_table(page_map),
        sanity_checks(doc_name, n_pages),
    ]
    return "\n\n".join(parts) + "\n"


def find_headings(
    paragraphs: list[dict], sections: list[dict], content: str
) -> list[Heading]:
    """Return every heading paragraph, in reading order, with both levels.

    How it works
        - Tree level: Azure puts a section's heading paragraph first in that
          section's `elements`. So walking the sections and reading each one's
          first element tells us which paragraph heads it, and the section's
          depth becomes that paragraph's level.
        - Markdown level: find the heading's line in `content` and count the
          leading '#'.
        - Reading order: sort by the heading's character offset in `content`,
          because `paragraphs` is not guaranteed to be in reading order.
    """
    depths = section_depths(sections)

    # paragraph index -> depth of the section it heads
    tree_level_of = {}
    for i, section in enumerate(sections):
        elements = section.get("elements", [])
        if elements and elements[0].startswith("/paragraphs/"):
            # "/paragraphs/12" -> 12
            paragraph_index = int(elements[0].split("/")[-1])
            tree_level_of[paragraph_index] = depths[i]

    headings = []
    for i, p in enumerate(paragraphs):
        if p.get("role") not in HEADING_ROLES:
            continue
        span = (p.get("spans") or [{}])[0]
        offset = span.get("offset", 0)
        headings.append(
            Heading(
                paragraph_index=i,
                # Collapse runs of whitespace and line breaks inside the heading.
                text=" ".join(p.get("content", "").split()),
                page=first_page(p),
                offset=offset,
                markdown_level=markdown_level(content, offset),
                tree_level=tree_level_of.get(i),
            )
        )
    return sorted(headings, key=lambda h: h.offset)


def section_depths(sections: list[dict]) -> list[int]:
    """Depth of every section in the tree. Section 0 is the root, depth 0.

    How it works
        Walk the sections in order. Whenever section i lists "/sections/j"
        among its elements, j is a child of i, so j's depth is i's depth + 1.

    Assumption, which the report will expose if wrong
        This needs every parent to appear in the list before its children,
        which Microsoft's sample does but no documentation guarantees. If it
        fails, children get depth 1 regardless of true nesting and the depth
        histogram in section 2 of the report will look implausibly flat.
    """
    depths = [0] * len(sections)
    for i, section in enumerate(sections):
        for element in section.get("elements", []):
            if element.startswith("/sections/"):
                child = int(element.split("/")[-1])
                depths[child] = depths[i] + 1
    return depths


def markdown_level(content: str, offset: int) -> int | None:
    """Count the '#' that open the markdown line containing `offset`.

    It is not documented whether a heading's span starts at its '#' or at the
    first word after it, so this looks at the whole line around the offset
    rather than at the offset itself -- either way the '#' are found.
    Returns None when the line has no '#' (Azure labelled it a heading but did
    not render it as one in the markdown).
    """
    line_start = content.rfind("\n", 0, offset) + 1
    line_end = content.find("\n", offset)
    # find() returns -1 when the heading is on the last line of the document.
    line = content[line_start : line_end if line_end != -1 else len(content)]
    hashes = len(line) - len(line.lstrip("#"))
    return hashes or None


def first_page(paragraph: dict) -> int:
    """The 1-indexed page a paragraph starts on.

    `boundingRegions` has one entry per page the paragraph touches, so the
    first entry is where it begins. Documented under "Figures" in the layout
    page as "including the page number". Returns 0 if Azure gave no region.
    """
    regions = paragraph.get("boundingRegions") or [{}]
    return regions[0].get("pageNumber", 0)


def build_page_map(
    headings: list[tuple[int, int, str]], n_pages: int
) -> dict[int, list[str]]:
    """Page -> the heading path in force on that page.

    This is the page-range attribution rule from `Build Order.md` section 1.3,
    and the answer to "how does page 13 know it belongs to a heading printed
    on page 12?": it never looks at page 13's text. The path is carried
    forward from earlier pages until a new heading replaces it.

    `headings` is (start_page, level, text) in reading order.

    How it works
        Keep a stack of the headings currently open, outermost first -- the
        stack IS the heading path. Walk pages 1..n. Before recording a page,
        apply every heading that starts on or before it:
          - pop every open heading at the same level or deeper, because a new
            heading closes its siblings and everything beneath them
          - push the new heading
        Then record the stack as that page's path.

        Worked example:
          Item 7 (level 1) starts p12  -> stack [Item 7]
          p13, p14 have no heading     -> stack unchanged, both get [Item 7]
          Results (level 2) starts p15 -> [Item 7, Results]
          Item 8 (level 1) starts p18  -> pops Results and Item 7 -> [Item 8]

    Several headings starting on one page
        All are applied before the page is recorded, so the page takes the
        path left after the LAST of them. A page that ends one section and
        starts another is therefore attributed to the new section.
    """
    stack: list[tuple[int, str]] = []  # (level, text), outermost first
    page_map = {}
    remaining = iter(headings)
    upcoming = next(remaining, None)  # the next heading not yet applied

    for page in range(1, n_pages + 1):
        # Apply every heading that has started by this page.
        while upcoming is not None and upcoming[0] <= page:
            _, level, text = upcoming
            # Close open headings at this level or deeper.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            upcoming = next(remaining, None)

        path = [text for _, text in stack]
        page_map[page] = path[:MAX_DEPTH]
    return page_map


# --- report sections ------------------------------------------------------
# Each function below renders one numbered section of the report as text.


def role_census(paragraphs: list[dict]) -> str:
    """Section 1: how many paragraphs carry each role.

    Shows the size of the stripping step -- what fraction of paragraphs are
    page headers, footers and page numbers that the chunker will discard.
    """
    counts = Counter(p.get("role") or "(body text)" for p in paragraphs)
    noise = sum(counts[r] for r in NOISE_ROLES)
    lines = [f"  {role:18} {n:6}" for role, n in counts.most_common()]
    share = noise / max(len(paragraphs), 1)  # max() guards an empty document
    return "\n".join(
        [
            "## 1. Role census",
            *lines,
            f"  -> {noise} of {len(paragraphs)} paragraphs are header/footer/"
            f"page-number noise to strip ({share:.0%})",
        ]
    )


def sections_summary(sections: list[dict], paragraphs: list[dict]) -> str:
    """Section 2: does Azure's sections tree actually nest?

    Reports how deep the tree goes, how many sections contain sub-sections,
    and whether sections carry `spans` (Microsoft's own sample shows them
    empty; if so, positions must come from the paragraphs instead). Then
    prints the first 40 sections indented by depth, so the shape of the tree
    can be seen directly.
    """
    depths = section_depths(sections)
    with_spans = sum(1 for s in sections if s.get("spans"))
    nested = sum(
        1
        for s in sections
        if any(e.startswith("/sections/") for e in s.get("elements", []))
    )
    spans_note = "   (EMPTY, as in Microsoft's sample)" if not with_spans else ""
    histogram = ", ".join(f"d{d}={n}" for d, n in sorted(Counter(depths).items()))
    lines = [
        "## 2. Sections tree",
        f"  sections: {len(sections)}   max depth: {max(depths, default=0)}   "
        f"containing sub-sections: {nested}",
        f"  sections with spans populated: {with_spans} of {len(sections)}"
        + spans_note,
        f"  depth histogram: {histogram}",
        "  first 40 sections, indented by depth (heading text, start page):",
    ]
    for i, section in enumerate(sections[:40]):
        label = section_label(section, paragraphs)
        lines.append(f"    {'  ' * depths[i]}[{i}] {label}")
    return "\n".join(lines)


def section_label(section: dict, paragraphs: list[dict]) -> str:
    """One-line description of a section: its first paragraph, page and role.

    If the first element is not a paragraph (the root usually starts with a
    sub-section), there is no heading text to show.
    """
    elements = section.get("elements", [])
    if not elements or not elements[0].startswith("/paragraphs/"):
        return "(no leading paragraph)"
    p = paragraphs[int(elements[0].split("/")[-1])]
    text = " ".join(p.get("content", "").split())[:70]
    return f"{text!r}  p{first_page(p)}  role={p.get('role') or 'body'}"


def heading_table(headings: list[Heading]) -> str:
    """Section 3: every heading, with its page and both measured levels.

    Two summary lines answer the spike's key questions:
      - do the tree level and the markdown level agree? Reported as the
        spread of (markdown '#' count - tree depth). If one value dominates,
        the two sources agree up to a constant and either could be used.
      - where do the "Item ..." headings sit? A 10-K's Items are its real top
        level, so they should all share one tree level.
    """
    # The relationship between the two levels is unknown until real output is
    # seen, so report the spread rather than assume an offset. (An earlier
    # version assumed '#' = depth + 1; the offline test showed that was wrong.)
    gaps = Counter(
        h.markdown_level - h.tree_level
        for h in headings
        if h.tree_level is not None and h.markdown_level is not None
    )
    gap_summary = (
        ", ".join(f"{g:+d} x{n}" for g, n in sorted(gaps.items())) or "none comparable"
    )
    items = [h for h in headings if h.text.lower().startswith("item ")]
    item_levels = sorted(Counter(h.tree_level for h in items).items(), key=str)

    lines = [
        "## 3. Headings",
        f"  headings found: {len(headings)}   with a tree level: "
        f"{sum(h.tree_level is not None for h in headings)}   with a markdown level: "
        f"{sum(h.markdown_level is not None for h in headings)}",
        f"  markdown '#' count minus tree depth, per heading: {gap_summary}",
        f"  'Item ...' headings: {len(items)}, at (tree level, count): {item_levels}",
        "  page  tree  md   text",
    ]
    for h in headings:
        lines.append(
            f"  {h.page:4}  {str(h.tree_level):>4}  {str(h.markdown_level):>3}   "
            f"{h.text[:80]}"
        )
    return "\n".join(lines)


def page_map_table(page_map: dict[int, list[str]]) -> str:
    """Section 4: every page with the heading path the chunker would attach.

    Shows both page numbers side by side, because the conversion between them
    is a known trap (`Systems Design Draft.md`, parser decisions): Azure's
    `pageNumber` is 1-indexed, FinanceBench's `evidence_page_num` is 0-indexed,
    so fb = azure - 1. The number printed in the filing's footer is neither
    and is never used.
    """
    lines = [
        "## 4. Page -> heading path",
        "  azure = Azure pageNumber (1-indexed); fb = FinanceBench evidence_page_num",
        "  azure   fb   heading path",
    ]
    for page, path in page_map.items():
        shown = " > ".join(path) or "(before any heading)"
        lines.append(f"  {page:5} {page - 1:4}   {shown}")
    return "\n".join(lines)


def sanity_checks(doc_name: str, n_pages: int) -> str:
    """Section 5: does Azure's page count match the PDF's own?

    A mismatch would mean Azure skipped or merged pages, and every page number
    in the map would then be suspect. PyMuPDF counts the PDF's pages directly.
    """
    import pymupdf

    pdf_pages = pymupdf.open(PDF_DIR / f"{doc_name}.pdf").page_count
    status = "OK" if pdf_pages == n_pages else "MISMATCH"
    return "\n".join(
        [
            "## 5. Sanity checks",
            f"  pages: Azure {n_pages}, PyMuPDF {pdf_pages} -> {status}",
            "  to check the index conversion: open the PDF at a page listed in",
            "  section 4 and confirm its heading; FinanceBench's page is the 'fb' column",
        ]
    )


if __name__ == "__main__":
    main()
