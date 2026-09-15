# RFI Discourse Corpus — Data Pipeline

Collection and transformation stage. Builds the corpus in `data/` from the
source databases; the viewer in `../viewer` reads only the output. Schema and
field semantics are documented in the repository `README.md`.

## Layout

```
pipeline/
├── scripts/
│   ├── 01_snapshot_ifp.py     one-time fetch of the IFP database
│   ├── 02_adapter_ifp.py      snapshot -> shared schema
│   ├── 03_profile.py          length-bias profile of the corpus
│   ├── 04_extract_fulltext.py NITRD PDFs -> data/fulltext/
│   └── 05_match_fulltext.py   segment vs. source -> source_match
└── data/
    ├── raw/                   snapshots (not committed)
    ├── dol_framework.json     DOL AI Literacy Framework, 12 units (context)
    ├── fulltext/              written by 04; one .txt per document
    ├── corpus.json            written by 02
    ├── documents.json         written by 02
    └── segments.json          written by 02
```

## Usage

```bash
python scripts/01_snapshot_ifp.py     # run once
python scripts/02_adapter_ifp.py --topic ""   # the whole corpus
python scripts/04_extract_fulltext.py         # submission text from the PDFs
python scripts/05_match_fulltext.py           # check each segment against its source
```

Run them in that order: 04 needs `documents.json` from 02, and 05 needs both. Re-running
02 rewrites `segments.json` and discards the verdicts, so 05 has to follow it.

`--topic "Talent & Education"` (or any substring of the IFP `Topic` tag) builds a
subset. `segment_id` is numbered over the whole record set before any filter is
applied, so an id names the same recommendation in every subset.

The adapter needs only the standard library. Extraction needs `pdftotext`
(poppler-utils) and the NITRD archive in `data/raw/`.

## Sources

| Layer | Source | Role |
|---|---|---|
| Index | [IFP AI Action Plan Database](https://www.aiactionplan.org) | sample selection via Topic tags |
| Text | [NITRD 90 FR 9088 responses](https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/) | primary text for analysis and release |
| Anchor | [America's AI Action Plan](https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf) (July 2025) | the policy output the submissions are read against |
| Context | DOL AI Literacy Framework (TEN 07-25) | a later, separately authorized document; interpretive context only |

## Notes

**Normalization happens once.** Annotation offsets are measured against the
output of `normalize()`. Changing that function invalidates every existing
offset. If it must change, name it `nfkc-v2` and write a remapping step. The
`text_sha256` on each segment is what detects this failure.

**IFP derived fields are unvalidated.** Both the text and the labels were
extracted by IFP using an LLM (stated on their
[methodology page](https://www.aiactionplan.org/methodology)). Use `Topic` only
as a sampling filter. Validate `Org Type`, or replace it with the official NITRD
respondent type, before using it as an analytic variable.

**Adapter contract:** `adapter(raw) -> {documents, segments}`. Downstream code
reads only the shared schema. A new corpus is one new adapter file.

## Known limitations

- `org_type_official` is `null` on every document. The NITRD respondent-type
  table has not been merged yet; until it is, `org_type_ifp` is the only
  organization label available and it is LLM-derived (see above).
- `dol_framework.json` has been transcribed from the source PDF but not yet
  verified line by line against it. It is context, so this is low priority.
- The anchor itself is not in this repository yet. Parsing America's AI Action Plan
  into the same schema (`layer: action-plan-recommendation`) is the next pipeline step.
- 1,513 segments (29%) carry `source_match: none`. Some of those are rewritten by the
  IFP extraction and some are quotations the PDF text is too damaged to match; the two
  are not yet told apart. Improving extraction would move some of them to `exact`.
- Four documents in `documents.json` point at files absent from the NITRD archive, and
  two more are scanned images. Six submissions therefore have no text.
- Segments carry `char_offset_in_doc: null`. Offsets into the submission PDFs
  require the full text, which is not part of this release.
