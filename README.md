# AI Action Plan RFI — a reader for 755 organizational submissions

In February 2025 the federal government asked the public what the United States
should do about AI ([90 FR 9088](https://www.federalregister.gov/documents/2025/02/06/2025-02305/request-for-information-on-the-development-of-an-artificial-intelligence-ai-action-plan)).
It received 10,067 responses. In July it published the AI Action Plan, with 103
actions in it.

This repository is a web-based reader for the 755 responses that came from
organizations, and for tagging what they ask for, paragraph by paragraph. It is
the first stage of a semester project: later stages embed those paragraphs and
compare them against the 103 actions, to ask which demands were taken up and
whose were not.

**DS 7400, Fall 2026 — HW1.** Work in progress.

---

## 1. Application Features and UI

Search across the top, the list of submissions on the
left, the selected submission on the right.

**Search** — the top area.

- A text box that matches organization names and topics as you type, and the
  full text of all 755 submissions when the backend is running.
- Two rows of filter chips, labelled `ORG TYPE` and `TOPIC`, each chip carrying
  the number of submissions it holds. Multi-select, and the two rows combine.

**List** — the left panel, one row per submission.

- Respondent type, then the organization's name, its topics, and how much there
  is: paragraph count and word count.
- The head of the panel says how many of the 755 are showing, and sorts them
  A–Z by organization or longest first by word count.
- A row shows a count once any of its paragraphs is tagged.

**Detail** — the right panel, the submission itself.

- The organization, its respondent type and a link to the original PDF on NITRD
  stay pinned at the top; the tag box stays pinned at the foot; only the text
  scrolls between them.
- The text is what was extracted from the PDF, split into numbered paragraphs.
  Click one to select it.
- Type a tag and press `Enter` to label the selected paragraph, or select a
  phrase inside it to label only that. A tagged paragraph carries a dot beside
  its number.

A badge in the header says where annotations are being written, `SQLite` or
`this browser`. A link at the foot exports every annotation as one JSON file.

## 2. The Data

US federal public records: comments submitted to OSTP in response to the RFI,
published by NITRD.

| | |
|---|---|
| Responses in the docket | 10,067 |
| Individuals — excluded | 9,312 |
| **Organizational submissions — the corpus** | **755** |
| Full text extracted | 752 (2 are scanned images, 1 is absent from NITRD's archive) |
| Paragraphs | 31,384 |
| Words | 1.97 million |

| File | What it is |
|---|---|
| `pipeline/data/documents.json` | 755 rows: organization, respondent type, date, source URL, topics |
| `pipeline/data/segments.json` | 31,384 paragraphs, each a pair of offsets into its document's text |
| `pipeline/data/fulltext/*.txt` | The extracted text the offsets point into, with a SHA-256 per file |
| `pipeline/data/corpus.json` | Provenance |

Retrieved 2026-09-12 · Unicode normalization `nfkc-v1` · offsets are code-point
offsets, counted the way Python's `len()` counts.

**Respondent type comes from the docket** — NITRD recorded it, so it is the
agency's own classification and not a model's guess. Private sector 291 ·
Non-profit 194 · Industry/professional/scientific association 178 · Academia 82 ·
Non-federal government 10.

**Topics do not.** They come from a third party, the
[IFP AI Action Plan Database](https://www.aiactionplan.org), which derived them
with a language model. They are unverified, so they are used to navigate the
corpus and never as an analysis variable.

A paragraph is stored as offsets rather than as a copy of the text, so the words
live in one place and cannot drift. The 603 MB source archive is not committed;
download it from [NITRD](https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/).

## 3. Running

Node.js 20+ and Python 3.11+.

**Viewer**

```bash
cd viewer
npm install
npm run dev
```

Open <http://localhost:5173>. This is all that is needed to read and tag.

**Backend** — optional, in a second terminal. It adds durable storage and
full-text search.

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The badge in the header changes from `annotations → this browser` to
`annotations → SQLite` once it connects. API docs: <http://localhost:8000/docs>.
A Dockerfile is included; see `backend/README.md`.

**Rebuilding the corpus** — not needed to run the app.

```bash
cd pipeline
python3 scripts/01b_snapshot_nitrd.py     # the respondent listing: the corpus spine
python3 scripts/02_adapter_ifp.py         # documents.json
python3 scripts/04_extract_fulltext.py    # needs pdftotext and the NITRD archive
python3 scripts/06_paragraphs.py          # segments.json
```

## 4. Libraries

| | |
|---|---|
| Frontend | React 19, Vite 8. No UI, state or routing library — the app is one screen |
| Backend | FastAPI, Uvicorn, Pydantic, SQLite (Python standard library) |
| Pipeline | Python standard library only, plus `pdftotext` (poppler-utils) |
| Fonts | IBM Plex Sans, IBM Plex Mono |


## 5. Annotation — implemented

The paragraph is the unit. A sentence is too small to carry a policy position
and a whole submission is too large to call one thing.

| | How | Stored as |
|---|---|---|
| A whole paragraph | Click it, type a tag, press `Enter` | `start: 0, end: <length>` |
| A phrase | Select text inside a paragraph; a box opens beside it | the selection's own offsets, with the quoted text |

`Enter` saves and moves to the next paragraph, so a pass runs down a document
without reaching for the mouse between paragraphs. Tags are free text with
autocomplete over the tags already used in that document, so `literacy` and
`AI literacy` do not accumulate as two names for one idea. A tagged paragraph
carries a dot beside its number.

Annotations are **stand-off**: an annotation records offsets into the text, and
never modifies it. The `scheme` field records which codebook a tag belongs to,
and `coder` records who made it — the field a second coder's pass will use.

## 6. Backend and database — implemented

FastAPI over SQLite, in `backend/`, with a Dockerfile.

| Table | Holds |
|---|---|
| `annotations` | One row per tag, in the stand-off shape above |
| `reviewed` | Paragraphs read and judged to contain nothing — a different fact from not having read them |

The viewer writes to the backend when it answers and to `localStorage` either
way, so a backend that goes down mid-session cannot take the work with it. The
client mints `ann_id`, so an annotation made while the backend was down keeps
its identity when it syncs and a replayed write cannot duplicate it.

`/search` reads the extracted text on disk and returns the submissions whose
bodies match. `/progress` reports how many paragraphs are tagged and which tags
have fewer than fifty examples, since a class that thin cannot be trained on.

---

## Upcoming Plan (can be changed)

| | |
|---|---|
| HW1 | This reader: read the corpus, tag paragraphs, store the tags |
| HW2 | Embed the 31,384 paragraphs and cluster them |
| HW3 | Train a classifier on the tags produced here |
| Final | Align the corpus against the AI Action Plan's 103 actions — which demands were taken up, and whose were not |

## Limitations

- **One coder.** The `coder` field exists on every annotation; the interface does not yet switch between coders.
- **Three submissions have no text.** Two are scanned images and need OCR; one is listed by NITRD but missing from NITRD's own archive.
- **Ten paragraphs exceed 2,000 characters** and would be truncated by a 512-token encoder. They need chunking before HW2.
- **88 paragraphs carry a replacement character** from a bad PDF font mapping — 0.004% of all characters.
- **The paragraph may not be the right unit.** Whether one paragraph holds one position is a question for HW2, answered by labelling a pilot rather than by argument. Because paragraphs are offsets, re-cutting them is one script re-run.

## Licenses

Code: MIT (`LICENSE`) · Data: CC BY 4.0 (`LICENSE-DATA.md`) · Citation: `CITATION.cff`
