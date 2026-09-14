# RFI Discourse Corpus — Data Pipeline

Collection and transformation stage for the DS 7400 semester project. See the
project document `rfi-corpus-spec.md` for the full research design and schema.

## Layout

```
rfi-pipeline/
├── scripts/
│   ├── 01_snapshot_ifp.py     one-time fetch of the IFP database
│   └── 02_adapter_ifp.py      snapshot -> shared schema
└── data/
    ├── raw/                   snapshots (not committed)
    ├── dol_framework.json     DOL AI Literacy Framework, 12 units
    ├── corpus.json            written by 02
    ├── documents.json         written by 02
    └── segments.json          written by 02
```

## Usage

```bash
python scripts/01_snapshot_ifp.py     # run once
python scripts/02_adapter_ifp.py      # Talent & Education subset
```

Pass `--topic ""` for all records. Standard library only; nothing to install.

## Sources

| Layer | Source | Role |
|---|---|---|
| Index | [IFP AI Action Plan Database](https://www.aiactionplan.org) | sample selection via Topic tags |
| Text | [NITRD 90 FR 9088 responses](https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/) | primary text for analysis and release |
| Anchor | DOL AI Literacy Framework | 12 query units for the alignment analysis |

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

## Next

- [ ] Add `data/raw/` to `.gitignore`
- [ ] Download the NITRD 600MB zip and keep it (processing waits for HW2; links rot)
- [ ] Fill `org_type_official` from the NITRD metadata table
- [ ] Verify `dol_framework.json` against the source PDF, then set `verified: true`
- [ ] Licenses (data CC BY 4.0, code MIT), `CITATION.cff`, Zenodo release
