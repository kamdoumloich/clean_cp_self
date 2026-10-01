#!/usr/bin/env bash

# 1. Default Symmetric Approach (THR)
./runWeightedRankBased.py --seed 2026 --dataset cifar10 --alpha 0.1 --n_bins 3 --max_rotation 179

# 2. Option 1: Global Agreement Count N_agree (APS)
# ./runWeightedRankBased.py --seed 2026 --dataset cifar10 --alpha 0.1 --n_bins 3 --approach agreement

# 3. Option 2a: Total Asymmetric Span W = d_L + d_R (APS)
# ./runWeightedRankBased.py --seed 2026 --dataset cifar10 --alpha 0.1 --n_bins 3 --approach asymmetric_span

# 4. Direct Side-by-Side Comparison of All Approaches
# ./runWeightedRankBased.py --seed 2026 --dataset cifar10 --alpha 0.1 --n_bins 3 --approach all
