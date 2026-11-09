# Manual error coding (CDCP dev, FT-all seed 0)

I coded 58 CDCP dev sentences to decide on the design of Experiment 2. The test set was never inspected.

Sentence text is omitted because CDCP is not redistributed. With CDCP downloaded
(`training/fetch_cdcp.py`), `hw3/diagnostics/error_sheet.py` and `error_sheet_d.py` rebuild the
sheet with text, joined on `id`.

## Groups (sampled with a fixed seed, 7)

| group | definition | rows / pool |
|---|---|---|
| A | gold policy, predicted another class | 9 / 9 |
| B | predicted policy, gold another class | 4 / 4 |
| C | gold policy, predicted policy (contrast) | 10 / 55 |
| D | gold fact or testimony predicted value (25 / 27), and gold value predicted fact or testimony (10 / 45) | 35 |

## Columns I coded

- `why_wrong`: free-text reason
- `ambiguous_to_me`: Y = I find the gold label ambiguous, N = I do not
- `notes`:
  - **Fragment**: cannot be judged without the surrounding comment
  - **None**: fits none of the four classes
  - **?label**: the model's prediction is wrong. My own label is in `why_wrong` when given
    (e.g., "It should be testimony"). In 2 rows (dev-24.1, dev-13.0) my label also differs from the gold label.

The other columns (gold, prediction, class probabilities, token count, modal position) come from the model.

Coding was done by a single coder and was not blind to the gold label or the prediction, so it serves as
qualitative motivation for the design rather than as evidence.
