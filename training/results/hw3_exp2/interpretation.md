## 7. Interpretation (untested)

_Everything in this section is interpretation and has not been tested._

**Headline: a null result.** No cue variant beats FT-all.
- **Test error.** FT-all 0.223 ± 0.007; CUE-GATE 0.227 ± 0.007, CUE-ONLY 0.226 ± 0.008, CUE-LIST 0.225 ± 0.009.
- **Paired by seed (same seed vs same seed).** Test error changes by −0.001 to +0.010. One test sentence is 0.0012.
- **Dev loss.** Higher than the same-seed FT-all in **9 of 9 runs**, by +0.004 to +0.014. The extra pooling path did not help on the data used for early stopping either.

**Predictions (Siwan, before the run)**
1. *"testimony weaker, policy stronger"*
   - Policy: **not supported**. Policy F1 is 0.814 for FT-all and 0.807–0.809 for the cue variants.
   - Testimony: CUE-GATE and CUE-ONLY move in the predicted direction, but within seed noise. Testimony recall goes 0.804 → 0.788 / 0.790 (std 0.022–0.026); seed 0 has 149 → 144 / 145 correct. CUE-LIST does not move (0.807).
2. *"CUE-GATE may be better"*: **not supported.** CUE-GATE ≈ CUE-ONLY ≈ CUE-LIST.
   - CUE-GATE ≈ CUE-ONLY because the learned gate stays high for every class (median g 0.81–0.97), so CUE-GATE behaves almost like g = 1.

**Why the extra path did not help (untested)**
- The design assumes the cue is diluted by mean pooling. After fine-tuning, that premise seems not to hold. Six layers of self-attention can spread "should" into every token vector, so the average already carries it.
- The occlusion check supports this. Deleting "should" collapses FT-all's P(policy) in 3 of 4 sentences (0.98 → 0.01, 0.99 → 0.20, 0.99 → 0.03), so mean pooling was not hiding the cue.
- The premise was more plausible for HW2's frozen encoder, which this experiment did not test.

**What attention found.** Learned attention (CUE-GATE / CUE-ONLY) puts its peak on:
- "should" in 62% of policy sentences;
- "i" in 18–25% of testimony sentences;
- auxiliaries and copulas ("is", "can", "are", "was") for fact and value.

One reading: it locates the main verb or modal of the clause. Using that vector directly did not change accuracy, which fits the information already being in m.

**CUE-LIST gate, counting only sentences that contain a list word** (median g; share with g > 0.5):

| gold | sentences with a list word | median g | g > 0.5 |
|---|---|---|---|
| policy | 70 | 0.914 | 94% |
| testimony | 121 | 0.122 | 0% |
| value | 119 | 0.188 | 10% |
| fact | 16 | 0.156 | 6% |

- The gate opens for policy sentences. Their list word is mostly "should".
- It stays shut for testimony sentences, even though 65% contain a list word; in 98% of those it is "i" or "my". The network learned that the vector of "i"/"my" adds nothing beyond m.
- The occlusion check points the same way: deleting "In my experience ," or "My experience is" barely moved FT-all.
- So testimony vs value seems to be decided by something other than first-person words. Past tense ("was" ranks 2nd for testimony) is one candidate, but untested.
- The summary gate table in §4 includes sentences without a list word; there c = m, so g has no effect.

**Near ties.** On the 20 near-tie sentences per seed, CUE-ONLY nets +6 and CUE-GATE +3 (summed over 3 seeds). Over all of dev both are net negative (−5, −8), so the near-tie gains are cancelled elsewhere. Twenty sentences per seed is too few to tell this from noise.

**Map.** FT-all and CUE-GATE show the same structure: policy and testimony form their own clusters, while fact and value overlap. Fact ↔ value remains the largest error:
- FT-all seed 0: 84 of 185 test errors (45%);
- CUE-GATE seed 0: 88.

These errors are not cue-word-shaped. Attention's top words for fact and value are generic auxiliaries.
