# Fase 2.0 -- A5 bilinear bug fix

**Diagnosis:** A5 already used balanced sample weights (`w_pos=|Y|/2n_pos`), so the imbalance hypothesis is not the cause. The defect was the hand-written fixed-step gradient descent, which did not converge. Fix: replace GD with **L-BFGS-B** (analytic gradient, convergence test, maxiter 5000), keeping balanced weights and rank 8, l2 1e-3.

## Loss (full training set, seed 0)

- BEFORE (GD, 300 iters): 0.6931 -> 0.6378
- AFTER (L-BFGS-B, 117 iters, converged=True): 0.6783 -> 0.4793
- full curve in `a5_loss_curve_lbfgs.csv`.

## Train / test MRR (5 seeds)

| | mrr_train | mrr_test |
|---|---|---|
| BEFORE (GD) | 0.1844 | 0.1498 |
| AFTER (L-BFGS-B) | 0.3868 | 0.1620 |

**Verdict:** A5 train MRR rose well above 0.184 -> the defect WAS the optimizer, not imbalance.
