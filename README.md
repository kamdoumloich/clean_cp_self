# Rotation-Invariant Ensemble Conformal Prediction

This repository implements an ensemble approach to conformal prediction for classification neural networks.

## Core Methodology

1. **Augmentation Invariance ($d$)**:
   - For an input $x$, we compute its rotation-invariance degree $d(x) \in [0, D_{\max}]$: the largest angle $d$ such that rotating $x$ by any degree $r \in [-d, d]$ preserves the model's top-1 predicted class.
   - The invariance domain $[0, D_{\max}]$ is partitioned into $B$ continuous ranges ("bins"), with each conformal predictor responsible for one bin.

2. **Precalibration Optimization**:
   - The calibration data is split into disjoint precalibration (e.g. 10%) and follow-up calibration (e.g. 40%) subsets.
   - An MILP optimization problem is formulated and solved on the precalibration set to jointly determine the bin boundaries $[L_b, R_b]$ and base nonconformity thresholds $\tau_b$ that minimize the sum of conformal set sizes subject to overall $(1 - \alpha)$ coverage.
   - The program explicitly computes and prints the coverage of each bin in the precalibration dataset for the computed thresholds and bin sizes.

3. **Follow-Up Calibration**:
   - On the follow-up calibration set, a single multiplicative factor $\lambda$ is computed such that scaled thresholds $\tau_b^{\text{final}} = \min(1.0, \lambda \cdot \tau_b)$ achieve the finite-sample $(1 - \alpha)$ marginal coverage guarantee.

4. **Inference & Benchmarking**:
   - Inputs are routed to their respective bin based on $d(x)$ and scored via $s(x, c) = 1 - p(c|x) \le \tau_{b(x)}^{\text{final}}$.
   - The approach is evaluated against standard conformal prediction (single global threshold) with overall metrics and a side-by-side per-bin breakdown on the test set.

---

## Environment Setup

The environment can be loaded using the conda environment at:
```bash
conda activate /home/lkd18/dev/miniconda3/envs/pytorch-cuda
```

---

## Running the Pipeline

### Quick Single Run
```bash
./single_run.sh
```

### End-to-End Command Line Execution
```bash
./runWeightedRankBased.py --seed 2026 --dataset cifar10 --alpha 0.1 --n_bins 3 --max_rotation 179
```

### CLI Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--seed` | int | `2026` | Random seed for data splitting and reproducibility |
| `--dataset` | str | `cifar10` | Dataset: `cifar10`, `cifar100`, `imagenet`, `mnist` |
| `--alpha` | float | `0.1` | Allowed error rate $\alpha$ (target coverage is $1 - \alpha$) |
| `--n_bins` | int | `3` | Number of invariance bins ($B$) |
| `--max_rotation` | int | `179` | Maximum rotation angle in degrees ($D_{\max}$) |
| `--min_samples_per_bin` | int | `20` | Minimum samples per bin in precalibration |
| `--train_fraction` | float | `1.0` | Fraction of training set to use |
| `--test_fraction` | float | `0.5` | Fraction of test set for testing |
| `--calibration_fraction` | float | `0.4` | Fraction of test set for follow-up calibration |
| `--precalibration_fraction` | float | `0.1` | Fraction of test set for precalibration |
| `--batch_size` | int | `256` | Inference batch size |
| `--approach` / `--routing_metric` | str | `symmetric` | Ensemble routing approach: `symmetric` (symmetric continuous degree $d \in [0, D_{\max}]$), `agreement` (Option 1: Global Agreement Count $N_{\text{agree}} \in [0, 2 D_{\max}]$), `asymmetric_span` (Option 2a: Total Asymmetric Span $W = d_L + d_R \in [0, 2 D_{\max}]$), or `all` (runs all three concurrently against baseline) |
| `--scoring_function` | str | `None` | Nonconformity scoring function: `thr` ($1 - p(y|x)$) or `aps` (cumulative probability). Defaults to `aps` for `agreement` and `asymmetric_span`, and `thr` for `symmetric` |
| `--score-in-log-space` | bool | `False` | Algorithmic mitigation for high coverage ($\alpha=0.01$ / 99% coverage): computes nonconformity scores in log-space ($s = -\log p$ for THR, $-\log(1 - s_{\text{aps}})$ for APS), eliminating the threshold saturation cliff at $\tau=1.0$ |
| `--coverage_slack` | float | `None` | Anti-cannibalization coverage slack $\epsilon$ (e.g. `0.02`). Each bin satisfies error rate $\le \alpha + \epsilon$ with at least 1 error allowed |
| `--min_bin_coverage` | float | `None` | Minimum precalibration coverage floor per bin |
| `--robust` | bool | `False` | Use robust trained weights if available |

---

### Running Alternative Rotation Approaches

#### Option 1: Global Agreement Count $N_{\text{agree}}$ (APS)
```bash
./runWeightedRankBased.py --dataset cifar10 --approach agreement --n_bins 3
```

#### Option 2a: Total Asymmetric Span $W = d_L + d_R$ (APS)
```bash
./runWeightedRankBased.py --dataset cifar10 --approach asymmetric_span --n_bins 3
```

#### Direct Side-by-Side Comparison of All Approaches
```bash
./runWeightedRankBased.py --dataset cifar10 --approach all --n_bins 3
```

#### ImageNet Evaluation
```bash
./runWeightedRankBased.py --dataset imagenet --alpha 0.1 --n_bins 3 --max_rotation 179
```

---

## Output & Reports

Runs generate:
1. **Precalibration Report**: Printed to stdout and saved in `logs/<dataset>/<seed>/precalibration_report_*.txt`.
2. **Test Comparison Report**: Side-by-side comparison of Ensemble vs. Standard Baseline Conformal Predictor across all bins saved in `logs/<dataset>/<seed>/test_bin_comparison_*.txt`.
3. **LaTeX Report**: Detailed report saved in `logs/<dataset>/<seed>/Report_*.tex`.
4. **Probability Cache**: Compressed `.npz` arrays in `logs/<dataset>/<seed>/` to make subsequent runs instantaneous and memory efficient (< 4 GB RAM).