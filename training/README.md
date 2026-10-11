# training/ — HW2

Proposition-type classifier (fact / policy / testimony / value), trained on
the public CDCP corpus and applied to the RFI paragraphs. Context and plan:
`../CLAUDE.md`.

## Setup (once)

```bash
cd ~/ai-corpus-discourse
python3 -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
python training/fetch_cdcp.py          # CDCP → training/data/cdcp/ (not committed)
python training/cdcp.py                # sanity check: counts per split
```

## Day 1 — go / no-go

```bash
python training/day1_compare.py                 # all-MiniLM-L6-v2 (downloads ~90 MB once)
python training/day1_compare.py --encoder mpnet # only if the first says NO-GO
```

Prints a table — shallow benchmark vs embedding + LogReg vs embedding + MLP —
and writes `training/results/day1_<encoder>.json`. Embeddings are cached in
`training/cache/`, so re-runs take seconds.

## Files

| | |
|---|---|
| `fetch_cdcp.py` | downloads CDCP (pinned commit of HITSZ-HLT/GMAM), writes a SHA-256 manifest |
| `cdcp.py` | loads CDCP as propositions; drops `reference` |
| `embed.py` | frozen encoders (`minilm`, `mpnet`, offline `lsa`), cached by encoder name |
| `day1_compare.py` | scouting comparison with scikit-learn — not the HW2 network |
| `model.py`, `train.py` | Day 2: the HW2 network and its own PyTorch training loop |

## Data

Park, J. & Cardie, C. (2018). A Corpus of eRulemaking User Comments for
Measuring Evaluability of Arguments. *LREC 2018.* JSON conversion with
train/dev/test splits from github.com/HITSZ-HLT/GMAM. Not redistributed here.
