## Short HOW-TO
- The environment I'm using for the simulation can be loaded using the file `conda env create -f environment.yml`
- The pattern based approach is written in the file `approaches/weightedRankBased.py`
- After activating the conda environment, the whole pipeline is executed using the following command
  - `python runWeightedRankBased.py [options]`

| Argument | Type | Required | Choices | Default | Description |
|--------|------|----------|--------|--------|-------------|
| `--seed` | int | Yes | - | - | Random seed for reproducibility |
| `--dataset` | str | Yes | `mnist`, `cifar` | - | Dataset to use for training and evaluation |
| `--robust` | bool | No | `True`, `False` | `False` | Whether to use robust training (note: type should be handled carefully in argparse) |
| `--alpha` | float | Yes | - | - | Alpha parameter for training (e.g., regularization strength) |
| `--min_samples_per_pattern` | int | No | - | `10` | Minimal number of samples per pattern (used in experiments) |
| `--train_fraction` | float | No | - | `0.1` | Fraction of training data to use |
| `--test_fraction` | float | No | - | `0.1` | Fraction of test data to use |
| `--calibration_fraction` | float | No | - | `0.2` | Fraction of data used for calibration |
| `--precalibration_fraction` | float | No | - | `0.01` | Fraction of data used for pre-calibration |