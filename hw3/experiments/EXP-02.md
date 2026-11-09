# EXP-02 — Cue-preserving pooling with a learned gate

Code: `training/cue_pool.py`, `training/hw3_exp2.py` · word list: `training/hw3_exp2_wordlist.txt` ·
results: `training/results/hw3_exp2/` (`report.md`, `interpretation.md`) · weights: `training/runs/hw3_exp2/` (gitignored)

## Hypothesis

FT-all averages all token vectors into one sentence vector (mean pooling). A short cue such as
"should", "please" or "In my experience" is one or two tokens out of many, so in the average it can
be outweighed by what the sentence is *about*. A second summary that can concentrate on a few
tokens (attention-weighted c), mixed in by a per-sentence gate g, should let those cues reach the
head. If so, sentences whose type turns on such a cue — especially those FT-all nearly ties on —
should be classified better, while the average still carries topic for sentences without one.

## Prediction

EXP-02 predictions (2026-11-07, before the run)
1. Testimony might not have enough cue words, so it will be detected less, but policy will be detected better.
2. CUE-GATE may be better, because the network sees the whole dataset, while my word list comes from a sample.

## Setup

- **Control = FT-all** (HW3, results reused, not re-run): CDCP train 2,742 / dev 525 / test 836;
  MiniLM (all-MiniLM-L6-v2) fine-tuned from pre-trained weights; AdamW, encoder lr 1e-5, head lr 1e-3,
  weight decay 0.01, 50-step linear warmup; early stopping on dev loss, patience 3, max 15 epochs;
  head 384 → 128 → ReLU → Dropout 0.2 → 4; L2 normalize before the head; seeds 0, 1, 2.
- **Only the pooling changes.** From the token vectors h_t (padding masked):
  m = mean_t h_t (incl. [CLS]/[SEP], = FT-all) · a_t = softmax_t(w · h_t) over word tokens
  (padding, [CLS], [SEP] excluded) · c = Σ a_t h_t · g = sigmoid(W_g [m; c] + b) ·
  z = g c + (1 − g) m → L2 normalize → head.
- **Variants:** CUE-GATE (learned w, learned g) · CUE-ONLY (learned w, g = 1) ·
  CUE-LIST (list-word tokens only, hard mean, c = m if none; learned g) · FT-all = g fixed to 0.
- **Word list** (case-insensitive, whole words): should, must, required, need to, needs to, please,
  important, I, my, experience.
- **Start:** w = 0, W_g = 0, b = 0 → uniform attention over word tokens and g = 0.5, so z is close to
  but not exactly m at epoch 0 (m includes [CLS]/[SEP], c does not). Head init, dropout masks and batch
  order equal FT-all's for the same seed. New pooling weights use the head lr (1e-3), wd 0.01.
- **Selection:** none. No grid; all hyperparameters inherited. Dev only for early stopping.
  Test scored once per variant × seed, at the end.
- **Checks before training:** (1) occlusion of cue phrases on FT-all seed 0 (dev); (2) the copied loop
  with g = 0 reproduces FT-all seed 0 epochs 0–2; (3) epoch-0 dev loss ≈ ln 4 and max |z − m| at start.

## Result

Run 2026-11-07, 9 runs, about 2.5 min each on CPU. Full numbers and figures are in
`training/results/hw3_exp2/report.md`.

**Checks**
- With g fixed at 0, the copied loop reproduces FT-all seed 0 (max diff 6e-8).
- Epoch-0 dev loss (seed 0): CUE-GATE 1.3907, CUE-ONLY 1.3909, CUE-LIST 1.3882. FT-all is 1.3905 and
  ln 4 = 1.3863.
- Max |z − m| at the start (seed 0, dev): 0.18, 0.30, 0.18. z starts near m, not equal to it.

**Test (CDCP, 836 sentences, mean ± std over seeds 0–2)**

| | HW2 | FT-all | CUE-GATE | CUE-ONLY | CUE-LIST |
|---|---|---|---|---|---|
| error | 0.292 ± 0.005 | **0.223 ± 0.007** | 0.227 ± 0.007 | 0.226 ± 0.008 | 0.225 ± 0.009 |
| macro-F1 | 0.649 | **0.751** | 0.745 | 0.747 | 0.748 |
| policy F1 | 0.614 | **0.814** | 0.808 | 0.807 | 0.809 |
| testimony F1 | 0.749 | **0.808** | 0.802 | 0.805 | **0.808** |
| testimony R | 0.761 | 0.804 | 0.788 | 0.790 | 0.807 |

- **Paired by seed.** Test error vs FT-all changes by −0.001 to +0.010. Dev loss is higher than FT-all's
  in 9 of 9 runs (+0.004 to +0.014).
- **Near ties on dev** (FT-all top-2 margin < 0.1; 20 per seed; summed over seeds):
  - fixed − broken: CUE-ONLY +6, CUE-GATE +3, CUE-LIST −2.
  - Over all of dev: −5, −8, −11.
- **Attention peak** (seed 0, test): "should" in 62% of policy sentences, "i" in 18–25% of testimony
  sentences, and "is" / "can" / "are" for fact and value.
- **Learned gate (CUE-GATE).** High for every class: median 0.81 (value) to 0.97 (policy).
- **CUE-LIST gate** (only sentences that contain a list word): opens for policy (median 0.91) and stays
  shut for testimony (median 0.12), whose matches are 98% "i"/"my".
- **Occlusion** (FT-all seed 0, before training):
  - Deleting "should" collapses P(policy) in 3 of 4 sentences.
  - Deleting "It's important to understand that" turns value into fact (P(fact) 0.08 → 0.84).
  - Deleting "In my experience" or "My experience is" leaves the predictions value; P(testimony) does
    not rise.

**Predictions**
1. Policy stronger: **no.** Testimony weaker: same direction for CUE-GATE and CUE-ONLY, but within seed
   noise.
2. CUE-GATE better: **no.** The three variants are indistinguishable.

## Interpretation

_Untested._
- **Null result.** Pooling is probably not the bottleneck once the encoder is fine-tuned. Self-attention
  can already spread a cue like "should" into every token vector, so the mean carries it. The occlusion
  check shows FT-all already reacts strongly to "should".
- The network itself rejected first-person words as a testimony cue: the CUE-LIST gate shuts on "i"/"my".
  This fits the occlusion result. Whatever separates testimony from value in FT-all, it is not those
  words.
- Fact ↔ value is still the largest error (45% of FT-all seed 0's test errors). Attention's top words for
  those classes are generic auxiliaries, so these errors do not look cue-word-shaped.

## Next

1. Report EXP-02 as a null result, with the mechanism evidence (occlusion, CUE-LIST gate). Stop the
   pooling line for this classifier.
2. Group D of the hand-coded error sheet is coded; see hw3/coding/.
3. Optional and cheap (no training): a paired per-sentence test (McNemar) of FT-all vs CUE-GATE on test,
   to state the null formally.
