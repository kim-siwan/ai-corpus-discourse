# AI Action Plan RFI — a reader for 755 organizational submissions

A web-based reader for the 755 organizational responses to the AI Action Plan
RFI ([90 FR 9088](https://www.federalregister.gov/documents/2025/02/06/2025-02305/request-for-information-on-the-development-of-an-artificial-intelligence-ai-action-plan)).
HW1 made them readable and taggable paragraph by paragraph. **HW2 adds a neural
network that labels every sentence as _policy_, _value_, _fact_ or
_testimony_**, so a reader can see where an organization asks the government
to act. **DS 7400, Fall 2026 — HW2.**

## 1. Application features and UI

**Corpus tab** — the HW1 reader: search and `ORG TYPE` / `TOPIC` filter chips
on top, the list of submissions on the left, the selected submission on the
right in numbered paragraphs, with the tag box at the foot (§8). New in HW2:

- **Predicted sentence types.** Opening a submission runs the network on all of
  its sentences (`GET /predict/document/{doc_id}`; usually well under a second).
  **Policy** sentences are shaded; hover any sentence for its type and
  probabilities. With the backend off, stored predictions are shown instead.
- **Show predicted types** toggle — hides all predictions, for hand-coding.

**Model tab**

- **Training monitor** — the served model's loss and error per epoch, and its train / dev / test results.
- **Interactive training** — pick a learning rate, epochs and class weight, press **Train** (§3).
- **Try the model on any text.** Paste text, press **Predict** (`POST
  /predict`): the sentences come back shaded, with a table of probabilities.
  This is for text outside the corpus — a sentence from the Action Plan
  itself, or one demand rephrased several ways — to see what the network
  responds to.

## 2. The model

| | |
|---|---|
| Unit | **one sentence** — a paragraph mixes types, a sentence has one |
| Input | the sentence's embedding from **all-MiniLM-L6-v2** (384 numbers), frozen |
| **Network** | `Linear(384 → 128) → ReLU → Dropout(0.2) → Linear(128 → 4)`, 49,796 weights |
| Training data | **CDCP** (Park & Cardie, 2018): citizen comments on a CFPB rule. Train 2,742 · dev 525 · test 836 |

**Why CDCP.** It is the closest public, gold-labeled corpus of comments
submitted to U.S. federal policymaking that I could find, with published
train / dev / test splits, and its policy label — a proposition proposing a
course of action — is what this project means by a demand. Its comments differ
from the RFI in who wrote them (citizens, not organizations) and what about
(debt collection, not AI); §4 measures what that difference costs.

**Training loop** (`training/train.py`, written out by hand): (1) initialize —
seed, Kaiming / Xavier weights, cross-entropy, plain SGD; (2) forward a batch
of 32; (3) loss; (4–5) `zero_grad()` and `loss.backward()`; (6) update
`w ← w − lr × gradient`; (7) repeat, keeping the epoch with the lowest dev loss
and stopping after 20 epochs without improvement. Settings were chosen on
**dev only** (`training/sweep.py`): lr 0.1, no class weighting. Test was scored once.

![Loss and error per epoch](training/artifacts/loss_curve.png)

| CDCP test (836 sentences) | Error | Macro-F1 | Policy F1 |
|---|---|---|---|
| Always predict `value` | 0.476 | — | — |
| **TF-IDF + logistic regression (benchmark)** | 0.327 | 0.615 | 0.71 |
| MiniLM + logistic regression | 0.291 | 0.640 | 0.59 |
| **MiniLM + our network** | **0.292 ± 0.005** (3 seeds) | 0.653\* | 0.62\* [0.52, 0.72] |

\* saved model (seed 0). Dev loss flattens around epoch 20 while train loss keeps
falling; early stopping kept epoch 37 (train error 0.206, dev error 0.263).

- **It beats the benchmark on error** by 3.5 points — but logistic regression on
  the same embeddings does as well: **the gain is the encoder's.**
- **It is weaker on policy:** 69% of missed policy sentences become `value`.
  MiniLM encodes what a sentence is about better than how it is said.

## 3. Interactive training and the monitor

`backend/train_api.py` serves the monitor and runs `train()` from `training/train.py`
in a background thread. Runs save to `training/runs/`; the served model is never
replaced and test is never scored.

## 4. On the RFI: unvalidated, and why

No RFI sentence has a checked label yet, so RFI predictions are **unvalidated demo
output**. What can be measured without labels says the transfer is poor:

| | MiniLM + our network | TF-IDF + LogReg |
|---|---|---|
| RFI sentences predicted policy (CDCP: 11–12%) | **5.2%** | 7.0% |
| *should / must / need to / require* sentences → policy, CDCP dev | 66% | 70% |
| — the same on the RFI | **19%** | **49%** |

Sources: `training/results/rfi_policy_share.json`, `rfi_transfer_tfidf.json`. The *should/must* comparison is a
heuristic, not accuracy. The sentences are picked by those words and TF-IDF
sees the words themselves, so its 49% partly reflects how the set was chosen.

- **The network misses most organizational demands.** *"The government should
  modernize FedRAMP by establishing a faster, criteria-based path for approval
  of AI tools."* (OpenAI submission) → value, policy 0.04.
- **The network drops more than the benchmark on this cue; a likely reason is
  that MiniLM weighs topic heavily and "should" lightly.**
- Lowering the threshold for "policy" would not fix it on its own: the missed
  demand above scores 0.04, and the predicted-policy set already holds noise
  such as a date line and footnote references.

## 5. The data

**RFI** — public records published by NITRD: 10,067 responses, **755 from
organizations**, 752 with extracted text; 31,384 paragraphs, 90,806 sentences.
`pipeline/data/` holds `documents.json`, `segments.json` (paragraphs as
offsets, in code points, `nfkc-v1`) and `fulltext/*.txt`. **Respondent type comes from
the docket. Topics** come from the [IFP database](https://www.aiactionplan.org), derived
by a language model: navigation only, **never fed to the model**. The 603 MB
source archive is on [NITRD](https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/).

**CDCP** — Park & Cardie (2018), *A Corpus of eRulemaking User Comments for
Measuring Evaluability of Arguments*, LREC. `training/fetch_cdcp.py` fetches it from
[HITSZ-HLT/GMAM](https://github.com/HITSZ-HLT/GMAM); it has no license, so **it is
not redistributed here**.

## 6. Running

Node.js 20+ and Python 3.11+. The viewer alone reads, tags and shows stored
predictions; live predictions and training need the backend, run from the
training environment (repository root, second terminal):

```bash
cd viewer && npm install && npm run dev     # http://localhost:5173
python3 -m venv training/.venv && training/.venv/bin/pip install -r training/requirements.txt
training/.venv/bin/uvicorn app:app --app-dir backend --port 8000
```

The HW1 backend (`backend/.venv` or its Dockerfile; see `backend/README.md`)
still runs everything from HW1; prediction and training then answer 503. To
reproduce the model (optional — the weights are committed; for the corpus itself
see `pipeline/README.md`):

```bash
training/.venv/bin/python training/fetch_cdcp.py      # CDCP (not committed)
training/.venv/bin/python training/sweep.py           # choose on dev, train, score test once
training/.venv/bin/python backend/predict_corpus.py   # every RFI sentence (~3 min)
```

## 7. Libraries

React 19 and Vite 8 (no UI, chart or state library) · FastAPI, Uvicorn,
Pydantic, SQLite · PyTorch, sentence-transformers
([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache-2.0), pysbd · scikit-learn, NumPy, Matplotlib · `pdftotext`.

## 8. Annotation and backend (HW1)

Click a paragraph, type a tag, press `Enter` to save and move on; or select a
phrase to tag only that. Annotations are **stand-off** (offsets, a `scheme`, a
`coder`), stored by FastAPI over SQLite and in `localStorage`. HW2's predictions
use the same shape; its endpoints live in new modules, and HW1's are unchanged.

## Limitations

- **Citizens → organizations:** RFI predictions are unvalidated and miss most demands (§4).
- **Unit mismatch:** CDCP is cut into clause-level propositions, the RFI into full sentences.
- **Small policy test set:** 93 sentences, so policy F1 is ±0.1.
- **PDF artifacts:** footnote markers leak into paragraphs ("1 of the art…"); three submissions have no text.

## AI use and code reference

**AI assistance:** This project was assisted with Claude Code.

- **Scripts & Documents** — On claude.ai I used Claude to discuss design
  decisions and to develop features presenting on the app. Under my direction,
  Claude Code wrote the training scripts and other codes needed for the
  project. README is also drafted by Claude, but reviewed manually.
- **What I decided:** the training data (CDCP), the research question, the unit
  of analysis, iteratively adjusting and choosing hyperparameters,
  decision-making considering final output of the project.
- Commits made with AI help carry a `Co-Authored-By: Claude` line.

**Borrowed code:** none copied from other projects. Libraries are used through
their public APIs (§7). The pretrained encoder is all-MiniLM-L6-v2 from
sentence-transformers; the CDCP data comes from the GMAM repository (§5).

## Licenses

Code: MIT (`LICENSE`) · Data: CC BY 4.0 (`LICENSE-DATA.md`) · Citation: `CITATION.cff` · CDCP and MiniLM keep their own terms.
