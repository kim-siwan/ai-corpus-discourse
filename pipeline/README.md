# Data pipeline

Builds the corpus in `data/` from the public record. The viewer in `../viewer`
reads only the output; it never sees the sources.

Nothing here is needed to run the application. The output is committed.

## The corpus in one line

**NITRD decides which submissions are in; the PDFs supply the text; IFP
supplies topic labels for navigation and nothing else.**

| Source | What it contributes | Trusted for analysis? |
|---|---|---|
| [NITRD respondent listing](https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/) | Which responses are organizational (755 of 10,067), and each one's respondent type | Yes — the agency's own record |
| NITRD PDF archive | The full text of each submission | Yes — it is the submission |
| [IFP AI Action Plan Database](https://www.aiactionplan.org) | 20 topic labels, and IFP's own recommendation records | No — LLM-derived, unverified |

## Scripts

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01b_snapshot_nitrd.py` | the NITRD listing | `raw/nitrd_responses_<date>.json` |
| 1 | `01_snapshot_ifp.py` | the IFP database | `raw/ifp_airtable_<date>.json` |
| 2 | `02_adapter_ifp.py` | both snapshots | `documents.json`, `ifp_recommendations.json`, `corpus.json` |
| 3 | `04_extract_fulltext.py` | `raw/NITRD-responses.zip` | `fulltext/<doc_id>.txt`, `fulltext/index.json` |
| 4 | `06_paragraphs.py` | `fulltext/` | **`segments.json`** — the paragraphs the viewer labels |
| — | `05_match_fulltext.py` | `ifp_recommendations.json`, `fulltext/` | adds a verdict to each IFP record |
| — | `03_profile.py` | `documents.json`, `segments.json` | a report on how text is distributed |

```bash
python3 scripts/01b_snapshot_nitrd.py
python3 scripts/01_snapshot_ifp.py
python3 scripts/02_adapter_ifp.py
python3 scripts/04_extract_fulltext.py   # needs pdftotext and the NITRD archive in data/raw/
python3 scripts/06_paragraphs.py
```

Steps 3 and 4 depend on step 2's `documents.json`. 05 and 03 are checks and
reports; the viewer does not need them. Everything runs on the Python standard
library except extraction, which needs `pdftotext` (poppler-utils).

The 603 MB archive and the two snapshots live in `data/raw/`, which is not
committed. Download the archive from NITRD.

## What is in `data/`

| File | Records | What it is |
|---|---|---|
| `documents.json` | 755 | One per organizational submission: organization, respondent type, source URL, IFP topics |
| `segments.json` | 31,384 | One per paragraph: `doc_id`, `char_offset_in_doc`, `char_end_in_doc`, `n_chars`, `n_words` |
| `fulltext/<doc_id>.txt` | 752 | The extracted text the offsets point into |
| `fulltext/index.json` | — | Per-document extraction status and SHA-256 |
| `ifp_recommendations.json` | 5,132 | IFP's segmentation of the same submissions, with `source_match` from 05 |
| `corpus.json` | 1 | Provenance |
| `dol_framework.json` | 12 | From an earlier framing of the project; unused |

## How paragraphs are cut

`06_paragraphs.py` splits each submission at blank lines, which is what
`pdftotext` emits between paragraphs. Runs under 80 characters are dropped:
page numbers, letterhead, stray headers. A block over 1,500 characters is cut
again at sentence ends, aiming for about 700, because some PDFs arrive with no
blank lines and would otherwise yield one 3,000-word paragraph.

A paragraph is stored as a pair of offsets into its document's text, not as a
copy. Readers slice: `text[char_offset_in_doc:char_end_in_doc]`. Offsets are
code-point offsets, counted the way Python's `len()` counts.

## Two things that must not drift

**Normalization happens once.** Text is normalized as `nfkc-v1`, defined in
`02_adapter_ifp.py` and imported by `04`. Every offset is measured against its
output. Changing it invalidates every offset and every annotation; if it must
change, call it `nfkc-v2` and write a remapping step.

**The text must not be re-extracted silently.** `fulltext/index.json` records a
SHA-256 of each document's text. A mismatch means that document's offsets are
stale.

## IFP's records

IFP's `Full text` field reads like a quotation and often is not. 05 checks
every record against its submission, ignoring case, whitespace and
punctuation:

| `source_match` | Records | |
|---|---|---|
| `exact` | 3,224 (62.8%) | Found in the submission |
| `partial` | 384 (7.5%) | Some lines found |
| `none` | 1,512 (29.5%) | Not found |
| `unavailable` | 12 (0.2%) | No text to check against |

`none` means not found, not invented: a real quotation goes undetected when a
PDF extracts badly. This is why the corpus is cut from the PDFs rather than
built from IFP's records.

## Known limitations

- **Three submissions have no text.** Two are scanned images and need OCR; one is listed by NITRD but missing from NITRD's own archive.
- **Ten paragraphs exceed 2,000 characters** (0.03%, holding 0.21% of the text). They need chunking before a 512-token encoder sees them.
- **88 paragraphs carry a replacement character** (U+FFFD) from a bad PDF font mapping.
- **The AI Action Plan is not here yet.** Parsing its 103 actions into the same schema is the next pipeline step.
