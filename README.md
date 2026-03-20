## Short HOW-TO
- The environment I'm using for the simulation can be loaded using the file `conda env create -f environment.yml`
- The pattern based approach is written in the file `approaches/weightedRankBased.py`
- After activation of the conda environment, the whole pipeline is executed using the following command
  - `python3 runWeightedRankBased.py [options]`
- A minimal test performed using the following command:
  - `./runWeightedRankBased.py --seed 2026 --dataset cifar --alpha 0.01 --train_fraction 1.0 --test_fraction 0.8 --precalibration_fraction 0.05 --min_samples_per_pattern 5`

| Argument | Type | Required | Choices | Default | Description                                                                      |
|--------|------|----------|--------|--------|----------------------------------------------------------------------------------|
| `--seed` | int | Yes | - | - | Random seed for reproducibility                                                  |
| `--dataset` | str | Yes | `mnist`, `cifar` | - | Dataset to use for training and evaluation (**BETTER USE CIFAR10**)              |
| `--robust` | bool | No | `True`, `False` | `False` | Whether to use robust training                                                   |
| `--alpha` | float | Yes | - | - | Alpha parameter for conformal prediction                                         |
| `--min_samples_per_pattern` | int | No | - | `10` | Minimal number of samples a valid pattern should have.                           |
| `--train_fraction` | float | No | - | `0.1` | Fraction of the original training set to use                                     |
| `--test_fraction` | float | No | - | `0.1` | Fraction of the original test set to use                                         |
| `--calibration_fraction` | float | No | - | `0.2` | Fraction of original test set used for calibration (disjoint from test_fraction) |
| `--precalibration_fraction` | float | No | - | `0.01` | Fraction of training data used for pre-calibration (subset of train_fraction)    |