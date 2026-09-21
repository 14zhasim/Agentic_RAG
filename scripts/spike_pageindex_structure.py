"""Compare PageIndex's document tree with Azure Document Intelligence's, on the same filing.

Why this exists
    Azure detects heading TEXT well but its LEVELS are unreliable: on 3M 2018 the
    21 Items land on four different levels, leaving 106 of 160 pages with no
    Item-level ancestor. The planned fix is an LLM pass over the heading list. If
    PageIndex produces a clean hierarchy straight away, that pass is unnecessary,
    so this measures the same four things on the same document.

Why it is run in an isolated environment, not added to the project
    PageIndex's local mode needs `litellm>=1.97`, which requires `openai<3.0.0`.
    This project pins `openai==3.8.0` for generation through OpenRouter, so adding
    PageIndex would downgrade the SDK the whole answer pipeline depends on. It is
    a one-off comparison, so it runs with `uv run --isolated --with` instead:

        set -a; source .env; set +a
        uv run --isolated --with "pageindex==0.2.15" \
            python scripts/spike_pageindex_structure.py index 3M_2018_10K --pages 20
        uv run --isolated --with "pageindex==0.2.15" \
            python scripts/spike_pageindex_structure.py analyse 3M_2018_10K

    Start with --pages 20 to check cost and behaviour on a slice before paying for
    the whole filing.

How PageIndex differs from Azure, which is the point of the comparison
    Azure gives character offsets into one markdown string, so text attaches to a
    heading exactly. PageIndex gives per-node PAGE indices, which is coarser. So
    even a better PageIndex hierarchy would need Azure's offsets for attribution —
    the interesting outcome is a hybrid: PageIndex for levels, Azure for text and
    offsets, joined on heading text.

Model routing
    Local mode indexes structure from layout statistics without an LLM; the index
    model only summarises and refines. litellm understands `openrouter/<model>`
    and reads OPENROUTER_API_KEY, so no OpenAI account is needed.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

PDF_DIR = Path("benchmarks/financebench/pdfs")
PARSED_DIR = Path("data/financebench/parsed")
STORAGE = PARSED_DIR / "pageindex_store"

# Same model the answer pipeline uses, so no new spending decision is introduced.
INDEX_MODEL = "openrouter/z-ai/glm-5.3-flash"

ITEM = re.compile(r"^item\s+\d+[A-Z]?\b", re.I)
PART = re.compile(r"^part\s+[IVX]+\b", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["index", "analyse"])
    parser.add_argument("doc_name")
    parser.add_argument(
        "--pages",
        type=int,
        default=0,
        help="index only the first N pages (0 = whole filing); use it to smoke-test cost",
    )
    args = parser.parse_args()

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "index":
        index(args.doc_name, args.pages)
    else:
        print(analyse(args.doc_name))


# --- step 1: index (costs LLM tokens) -------------------------------------


def index(doc_name: str, pages: int) -> None:
    """Build PageIndex's tree locally and cache it next to the Azure JSON.

    Local mode is synchronous: `submit_document` returns once indexing is done.
    The tree is then fetched with summaries off, since only the hierarchy is
    being compared.
    """
    out = PARSED_DIR / f"{doc_name}.pageindex.json"
    if out.exists():
        print(f"{doc_name}: cached at {out}, skipping")
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set. Run: set -a; source .env; set +a")

    pdf = source_pdf(doc_name, pages)
    from pageindex import PageIndexClient

    client = PageIndexClient(
        mode="local",
        index_model=INDEX_MODEL,
        storage_path=str(STORAGE),
    )
    print(f"{doc_name}: indexing {pdf} locally with {INDEX_MODEL} ...")
    result = client.submit_document(str(pdf))
    doc_id = result["doc_id"]
    tree = client.get_tree(doc_id, node_summary=False)

    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(tree))
    tmp.rename(out)
    print(f"{doc_name}: tree cached at {out}")


def source_pdf(doc_name: str, pages: int) -> Path:
    """The filing, or a truncated copy of its first `pages` pages for a cheap trial."""
    pdf = PDF_DIR / f"{doc_name}.pdf"
    if not pdf.exists():
        sys.exit(f"{doc_name}: no PDF at {pdf}")
    if not pages:
        return pdf

    import pypdfium2  # ships with pageindex, so no extra dependency

    cut = PARSED_DIR / f"{doc_name}.first{pages}.pdf"
    if not cut.exists():
        src = pypdfium2.PdfDocument(str(pdf))
        dst = pypdfium2.PdfDocument.new()
        dst.import_pages(src, list(range(min(pages, len(src)))))
        dst.save(str(cut))
    return cut


# --- step 2: analyse (free, offline) --------------------------------------


def analyse(doc_name: str) -> str:
    """Report the same four measures taken on Azure's output, for comparison."""
    path = PARSED_DIR / f"{doc_name}.pageindex.json"
    if not path.exists():
        sys.exit(f"{doc_name}: not indexed yet. Run the index step first.")
    tree = json.loads(path.read_text())

    nodes = flatten(tree)
    items = [n for n in nodes if ITEM.match(n["title"])]
    parts = [n for n in nodes if PART.match(n["title"])]
    levels = Counter(n["level"] for n in items)
    depths = Counter(n["level"] for n in nodes)
    suspicious = [n for n in nodes if split_word(n["title"])]

    lines = [
        f"# PageIndex structure report: {doc_name}",
        "",
        f"nodes: {len(nodes)}   max depth: {max(depths, default=0)}",
        f"depth histogram: {dict(sorted(depths.items()))}",
        "",
        "## 1. Are the Items siblings? (Azure: no, spread over 4 levels)",
        f"  'Item ...' nodes: {len(items)}   levels: {dict(sorted(levels.items()))}",
        f"  -> {'CLEAN' if len(levels) == 1 else 'INCONSISTENT'}",
        "",
        "## 2. Are the Parts present? (Azure: PART I absorbed into the previous heading)",
        *[f"  level {n['level']}  p{n['start']}  {n['title'][:60]}" for n in parts],
        f"  -> {len(parts)} of 4 Parts found",
        "",
        "## 3. Heading text fidelity (Azure: 16 of 295 had split-word typos)",
        f"  nodes with a suspicious 1-2 letter fragment: {len(suspicious)} of {len(nodes)}",
        *[f"    {n['title'][:70]}" for n in suspicious[:8]],
        "",
        "## 4. The Items, with their start pages",
        *[f"  level {n['level']}  p{n['start']:<5} {n['title'][:66]}" for n in items],
    ]
    return "\n".join(lines)


def flatten(tree: dict, level: int = 0) -> list[dict]:
    """Walk PageIndex's nested nodes into a flat list of (title, level, start page).

    The payload nests children under a "nodes" key; titles and page indices vary
    slightly by version, so several key spellings are accepted rather than
    assuming one.
    """
    out = []
    children = tree.get("nodes") or tree.get("children") or []
    for node in children:
        title = str(
            node.get("title") or node.get("node_title") or node.get("text", "")
        ).strip()
        start = (
            node.get("start_index")
            or node.get("page_index")
            or node.get("start_page")
            or node.get("physical_index")
            or 0
        )
        out.append({"title": " ".join(title.split()), "level": level, "start": start})
        out.extend(flatten(node, level + 1))
    return out


REAL_SHORT_WORDS = {
    "a", "i", "in", "of", "to", "on", "is", "as", "at", "or",
    "an", "by", "be", "it", "no", "we", "us", "up", "do", "if",
}


def split_word(title: str) -> bool:
    """True if the title contains a stray 1-2 letter fragment, e.g. 'Busines s.'."""
    for word in title.split():
        bare = word.strip(".,:;'").lower()
        if bare.isalpha() and 1 <= len(bare) <= 2 and bare not in REAL_SHORT_WORDS:
            return True
    return False


if __name__ == "__main__":
    main()
