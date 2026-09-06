# Systems Design Draft Notes

\--

Retrieving files directly

<https://www.sec.gov/answers/reada10k.htm>

also check downloaded pdf file

Yes. If your goal is **SEC filings and learning how companies actually
tag financial statements**, I'd focus on **XBRL US + SEC + XBRL
International**, in that order.

This site (<https://data.sec.gov>) provides public access to the U.S.
Securities and Exchange Commission\'s EDGAR data via Application
Programming Interfaces (API). Automated access must comply with
SEC.gov\'s Privacy and Security Policy.

Please visit <https://www.sec.gov/edgar/sec-api-documentation> for
documentation. More developer resources and Fair Access guidelines can
be found at <https://www.sec.gov/developer>

<https://www.sec.gov/search-filings/edgar-application-programming-interfaces>

\-\--

**[Retrieving Filings and their Metadata]**

**Identifiers:**

- Accession number: identifies one specific filing (e.g.
  0000320193-23-000106), dashes stripped for the URL

**CIK → accession number (and other important metadata)):**

1.  CIK: SEC\'s company ID, zero-padded to 10 digits (Apple: 0000320193)

2.  Ticker → CIK via sec.gov/files/company_tickers.json

3.  CIK → accession number via
    data.sec.gov/submissions/CIK##########.json. It returns v useful
    fields such as
    1.  **cik** --- the company identifier, stable for the company\'s
        life

    2.  **accessionNumber** --- unique ID for this specific filing

    3.  **formType/form** --- \"10-K\", \"10-Q\", \"8-K\", etc.
        (amendments get a /A suffix, e.g. \"10-K/A\")

    4.  filter parallel arrays for form == \"10-K\", read
        accessionNumber + primaryDocument at that index

    5.  **reportDate** --- the actual fiscal period end this filing
        covers (e.g. a 10-K filed in November 2023 might have reportDate
        of 2023-09-30, the fiscal year-end, not the filing date). **This
        is the field you actually want when matching a question like
        \"what was revenue in FY2023,\" not filingDate.**

    6.  **primaryDocument** --- the filename you use to construct the
        actual document URL

    7.  **isXBRL** --- boolean, whether this filing has inline XBRL

**HTML, not PDF:**

- 10-Ks are HTML, tables have real \<table\>/\<tr\>/\<td\> structure
  already, unlike PDF

- Since 2019, XBRL is embedded directly in that same HTML as inline
  XBRL (iXBRL)

**XBRL:** tags specific numbers with standardized concepts (e.g.
us-gaap:Revenues) so they\'re comparable across companies/filings -- all
companies mandates from 2009-11

**Table formatting:**

Table

↓

row labels

↓

column headers

↓

footnote

↓

preceding paragraph

**Pipeline pattern to adopt:** fetch → cache to disk → transform/embed →
store

**Parsing methods investigated:**

- sec-parser: unmaintained, 10-K secondary. Not recommended.

- Unstructured.io: PDF-oriented, dependency friction.

- BeautifulSoup classifier: table/text fine, heading detection
  unresolved (bold via inline CSS, not \<b\>/\<strong\>).

May just opt for parsing pdfs instead of html, given that the former is
way more applicable to industry practice than working with html
reporting e.g. financial reports of private companies.

\-\--

**The big one: every 10-K follows the exact same fixed structure, by
law**

Every 10-K, from any company, is organized into the same four Parts with
the same numbered Items in the same order, because the SEC mandates this
template. You can build heading-detection logic around matching these
specific Item numbers/names as known anchors, rather than relying purely
on styling heuristics (bold, font-size) that you\'ve seen already vary
unpredictably between filers.

**The full structure, with what\'s actually useful vs. low-value:**

**Part I** --- mostly qualitative, describes the business

- Item 1, Business --- company overview

- Item 1A, Risk Factors --- high value for your project; explicitly
  the risk-factor content your Phase 3 list flagged

- Item 1B, Unresolved Staff Comments --- almost always
  boilerplate/empty

- Item 1C, Cybersecurity --- newer item, company-specific

- Item 2, Properties --- usually low information density

- Item 3, Legal Proceedings --- can matter for qualitative questions

- Item 4, Mine Safety Disclosures --- near-universally irrelevant
  unless mining company

**Part II** --- this is where your actual numbers live

- Item 5, Market for Registrant\'s Common Equity --- stock/equity
  data, moderate value

- **Item 6 is explicitly \"\[Reserved\]\"** --- literally nothing
  here, don\'t waste parser effort looking for content

- **Item 7, MD&A** --- your highest-value section. This is where
  management explains _why_ the numbers changed, exactly the
  qualitative reasoning layer that sits on top of the raw financial
  statements

- Item 7A, Quantitative and Qualitative Disclosures About Market Risk
  --- often short, sometimes skipped entirely (see exemption note
  below)

- **Item 8, Financial Statements and Supplementary Data** --- the
  actual financial statements (income statement, balance sheet, cash
  flow) plus their footnotes. This is your other highest-value
  section, and it\'s exactly where the table→footnote relationship
  problem lives

- Item 9 / 9A / 9B / 9C --- almost entirely boilerplate/compliance
  disclosures, low retrieval value

**Part III** --- often not actually present in the document

- Items 10--14 (directors, executive compensation, ownership,
  related-party transactions, accountant fees - respectively).
  **Important gotcha**: the form explicitly allows companies to
  incorporate this whole Part by reference from their proxy statement
  (DEF 14A) instead of writing it into the 10-K itself.

**Part IV** --- pure filing mechanics

- Item 15, Exhibits and Financial Statement Schedules --- mostly a
  list of attached exhibit files

- Item 16, Form 10-K Summary --- optional, most companies skip it
  entirely

- Signatures --- boilerplate, ignorable for retrieval purposes

**Other structural quirks worth knowing:**

- **The cover page is a wall of yes/no checkboxes** (large accelerated
  filer? shell company? etc.), filer-classification metadata:
  low-priority noise in your parser

- **\"Smaller reporting companies\" get exemptions**: they can skip
  Item 7A (market risk) and Item 1A (risk factors) entirely.

- **Item 6 being \"\[Reserved\]\"** is a leftover from an old
  requirement (previously \"Selected Financial Data\") that got
  removed but the number was kept empty

**Practically, for your parser:** the single highest-leverage move here
is building a lookup list of the known Item headings (\"Item 1.\",
\"Item 1A.\", \... \"Item 16.\") and using them as reliable
section-boundary anchors, on top of or instead of your current
bold-styling heuristic. Since this structure is legally mandated and
identical across every filer, it\'s a far more robust signal than
inferring headings from inconsistent CSS styling

---

Run benchmark at every change:

LlamaIndex to orchestrate pipeline

Ingest files

- Load file
- Chunk + metadata attribution / structure parsing
  - different methods for keeping track of document structure to attach as metadata to a chunk (HiChunk)
  - SLM to generate heading to attach as section heading to metadata (Fin-STAR)
  - table-aware chunking
- Build index for keyword search
- Embed chunk
  - figure out best embedder model for this

Retrieve - Elastic search

- Metadata filtering
- Hybrid search
  - BM25
  - Semantic search
  - rrf
  - tune top-k

Reranking – cross encoder

Agentic tooling

- choose BM25 or semantic search + specific queries to use for each
- query enhancement
- query decomposition (core question)
- HyDE (fake answer)
- feedback loop: multi-hop retrieval
- calculator tool
- verification agent (compare retrieved info vs generated answer)

LLM (openai sdk) – answer generation
