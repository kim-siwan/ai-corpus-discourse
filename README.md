# AI Action Plan RFI — a discourse corpus and reader

What did organizations ask the federal government for, and what made it into policy?

The Federal Register notice [90 FR 9088](https://www.federalregister.gov/documents/2025/02/06/2025-02305/request-for-information-on-the-development-of-an-artificial-intelligence-ai-action-plan)
states that its purpose was to inform an AI Action Plan. That plan was published in
July 2025, which makes the pairing an administrative record rather than an inference:
the submissions are the input and the plan is the output.

This repository holds 5,166 policy recommendations drawn from 751 of those
submissions, in a documented schema, together with a browser-based reader for working
through them and tagging what they say. The reader is the first piece of a
semester-long project: later stages embed the corpus, cluster it, and train a
classifier on the tags produced here.

**Status:** work in progress (DS 7400, Fall 2026).

---

## What the application does

A two-panel reader. The left panel lists recommendations; the right shows the one
selected, in the context of the submission it came from.

| Feature | What it does |
|---|---|
| **Browse** | All 5,166 segments in one scrolling list: id, organization, organization type, title |
| **Select** | Click a row, or press `J` / `K` to move without touching the mouse |
| **Search** | Matches the title, the organization name **and the recommendation text**. Hits are highlighted, and a row shows the matching passage when the hit is in the body rather than the title |
| **Filter** | Ten organization-type chips, multi-select. A count of what is showing |
| **Read the source** | *Read more* opens the full submission — a median of 18,000 characters against the excerpt's 788 — fetched on demand and cached |
| **Move within a document** | 150 documents hold more than one recommendation; the detail panel lists the siblings and moves between them |
| **See what is trustworthy** | Every segment is checked against its source PDF. Unverifiable ones are hidden by default behind an `Unverified` toggle, and partial matches carry a warning line |
| **Track progress** | A `Reviewed` flag per segment, and an `Untagged only` filter for a pass spread over several sittings |
| **Export** | All annotations as stand-off JSON |

The viewer follows the system colour scheme and works down to phone width.

### Annotation — yes, implemented

Two paths, because a labelling pass needs both:

| Path | How | Stored as |
|---|---|---|
| **Whole segment** | Type in the field pinned to the foot of the panel, press `Enter` | `start: 0, end: <length>` |
| **A phrase** | Select text in the passage; a small box opens beside the selection | the selection's real offsets, with the quoted text |

Tags are free text with autocomplete over tags already used, so `literacy` and
`AI literacy` do not accumulate as separate names for one idea. A fixed codebook is
deliberately absent: it is meant to emerge from the data rather than be imposed now,
and the `scheme` field on every annotation records which codebook a tag belongs to
when one exists.

The keyboard flow is the point. `J` and `K` move, the tag field takes focus as each
segment opens, and `Enter` saves and advances. At thirty seconds a segment a thousand
of them take eight hours; at ten seconds they take under three.

Offsets are code-point offsets into `segment.text`, counted the way Python's `len()`
counts, so annotations line up with the analysis code rather than with JavaScript's
UTF-16 string indices.

### Backend and database — yes, implemented

FastAPI over SQLite (`backend/`), with a Dockerfile.

| Table | Holds |
|---|---|
| `annotations` | One row per tag, in the stand-off shape above |
| `reviewed` | Segments read and judged to contain nothing — a different fact from not having read them, and one that cannot be recovered afterwards |

The viewer writes to the backend when it answers and to `localStorage` when it does
not, and to `localStorage` either way; a header badge says which store is live. The
client generates `ann_id`, so an annotation made while the backend is down keeps its
identity when it syncs and a replayed write cannot duplicate it.

`GET /progress` reports how many segments are tagged and which tags have fewer than
fifty examples, since a class that thin cannot be trained on. See `backend/README.md`.

---

## The data

Public comments submitted to the Office of Science and Technology Policy in March
2025, in response to a Federal Register request for information on an AI Action Plan.
They are US federal public records.

| File | Records | What it is |
|---|---|---|
| `documents.json` | 751 | One per submission: organization, type, date, source URL |
| `segments.json` | 5,166 | One per recommendation. `layer = ifp-recommendation` |
| — verified against source | 3,232 | `source_match = exact`, with `char_offset_in_doc` filled |
| `fulltext/<doc_id>.txt` | 745 | Full submission text from the PDFs |
| `dol_framework.json` | 12 | DOL AI Literacy Framework — interpretive context, not an anchor |
| `corpus.json` | 1 | Corpus- and layer-level provenance |

**Retrieved** 2026-09-12 · **Normalization** `nfkc-v1` · **Integrity** SHA-256 per segment

### What counts as a segment

A segment is one recommendation as extracted by the
[IFP AI Action Plan Database](https://www.aiactionplan.org), not a unit this project
defined. Measured against the source PDFs, 90% of excerpts begin at a line start or a
bullet, so the unit is a structural one — a bullet, or a heading with the paragraph
under it — and excerpts cover a median 20% of the submission they come from. The
remaining 80% is cover letter, background, argument and signature.

The boundaries were chosen by `claude-3-7-sonnet-latest`. Any analysis built on these
segments inherits that segmentation.

### What is and is not in the corpus

The RFI drew **10,068 submissions**. This corpus holds the **751** that IFP indexed
and attributed to an organization — roughly 7.5%. IFP does not publish its inclusion
rule, so the corpus should be described as *submissions IFP indexed*, not as *all
organizational submissions*. Individual submissions are present in the NITRD archive
and are not yet processed.

A handful of indexed submissions carry an individual's own name as the organization,
so the organizational boundary is not sharp.

### Whether a segment is a quotation

IFP's `Full text` field reads like a quotation and is not consistently one. Every
segment is matched against its submission, ignoring case, whitespace and punctuation
(`pipeline/scripts/05_match_fulltext.py`):

| `source_match` | Segments | Meaning |
|---|---|---|
| `exact` | 3,232 (63%) | Found in the submission; `char_offset_in_doc` filled |
| `partial` | 391 (8%) | Some lines found, some not |
| `none` | 1,513 (29%) | Not found |
| `unavailable` | 30 (1%) | No text could be extracted, so no check was possible |

`none` means **not found**, not **invented**. A real quotation goes undetected when a
PDF extracts badly, and inspection shows both causes present. The viewer hides these
by default.

This distinction is not decoration. The project asks who demanded what, and a
rewritten passage carries the rewriter's modality rather than the organization's — the
difference between *should support* and *must be a core requirement* is the
measurement. A corpus that mixes quotation with paraphrase and marks neither cannot
support the claim.

### Other derived fields

- `org_type_ifp` is LLM-generated and **is not used as an analysis variable**. `org_type_official` is reserved for the NITRD-declared respondent type.
- The 603 MB source archive (`pipeline/data/raw/`) is not committed. Download it from [NITRD](https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/).

### On the DOL AI Literacy Framework

An earlier version of this project treated the framework as an outcome of the RFI.
That was wrong. [TEN 07-25](https://www.dol.gov/agencies/eta/advisories/ten-07-25)
states its own basis as Executive Order 14277 on AI education for American youth, and
cites neither the RFI nor the OSTP process. It is kept as interpretive context.

---

## Running the application

Requires Node.js 20+ and Python 3.11+. The viewer runs on its own; the backend adds
durable storage for annotations.

**Viewer**

```bash
cd viewer
npm install
npm run dev
```

Open the URL printed in the terminal, usually <http://localhost:5173>.

**Backend** (optional, in a second terminal)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The header badge changes from `annotations → this browser` to
`annotations → SQLite` once it connects. API docs are at
<http://localhost:8000/docs>. A Dockerfile is included; see `backend/README.md`.

**Rebuilding the corpus** (not needed to run the app)

```bash
cd pipeline
python3 scripts/01_snapshot_ifp.py          # once
python3 scripts/02_adapter_ifp.py --topic ""
python3 scripts/04_extract_fulltext.py      # needs pdftotext and the NITRD archive
python3 scripts/05_match_fulltext.py
```

---

## Built with

| | |
|---|---|
| Frontend | React 19, Vite 8. No UI, state or routing library — the app is one screen |
| Backend | FastAPI, Uvicorn, Pydantic, SQLite (Python standard library) |
| Pipeline | Python standard library only, plus `pdftotext` (poppler-utils) for extraction |
| Fonts | IBM Plex Sans and IBM Plex Mono |

---

## Repository layout

| Path | What it is |
|---|---|
| `pipeline/` | Scripts that build the corpus from source documents, and the corpus itself |
| `viewer/` | React + Vite reader and annotation interface |
| `backend/` | FastAPI + SQLite annotation store |

The pipeline is organised around one contract: `adapter(raw) -> {documents, segments}`.
Everything downstream reads only that schema and does not know the source, so a new
corpus is one new adapter file.

---

## Known limitations

- **Annotation works on the excerpt, not the full submission.** Offsets are measured against `segment.text`; tagging a phrase in the full text needs document-level offsets, which is a schema change rather than a UI one.
- **1,513 segments are unverified** and hidden by default. Improving PDF extraction would move some of them to `exact`.
- **Six submissions have no text**: two are scanned images, four are absent from the NITRD archive.
- **Long segments are not chunked.** 105 segments exceed 2,000 characters and hold 39% of all text; embedding them whole would truncate at the encoder's limit.
- **One coder.** The `coder` field exists on every annotation, but the interface does not yet switch between coders.

---

## Licenses

- Code: MIT (`LICENSE`)
- Data: CC BY 4.0 (`LICENSE-DATA.md`)

## Citation

See `CITATION.cff`.
