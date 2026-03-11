#!/usr/bin/env python3
# An example for running for COMPACT

import compact
import benchmarks.mnist
import benchmarks.cifar
import approaches.base
import approaches.weightedRankBased
import interestingcasefilters.largedeviation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True, choices=['mnist', 'cifar'])
args = parser.parse_args()

seed_nber = args.seed
dataset = args.dataset

dict_augmentations = {
    "ORIGINAL": 0,
    "ROTATION": 10,
}

alpha = 0.01
# alpha = 0.1

is_robust = True


if dataset == 'cifar':
    raise NotImplementedError("Robust training for CIFAR10 is not implemented yet.")

    latex_file_name = f"./latex/cifar/ClassWeightedRankBased{seed_nber}Robust{is_robust}.tex"

    mnistBenchmark = benchmarks.cifar.CIFARBenchmark(augmentations=dict_augmentations, seed=seed_nber,
                                                     robust_training=is_robust)
    n_classes = 10
elif dataset == 'mnist':
    latex_file_name = f"./latex/mnist/ClassWeightedRankBased{seed_nber}Robust{is_robust}.tex"

    mnistBenchmark = benchmarks.mnist.MNISTBenchmark(augmentations=dict_augmentations, seed=seed_nber,
                                                     robust_training=is_robust) #2025, 9009, 777, 42
    n_classes = 10

else:
    raise NotImplementedError

# assert False
baseApproach = approaches.base.MonotoneConformanceEvaluatorSingleDistributionAllAboveThresholdButAtLeastOneClass()
weightedApproach = approaches.weightedRankBased.WeightedRank2DPredictor(precalibration_data=mnistBenchmark.precalibration_data,
                                                                        seed=seed_nber, n_classes=n_classes,
                                                                        )

# ayratsApproach = approaches.twoD.Ayrats2DPredictor()
interestingCaseFilter = interestingcasefilters.largedeviation.LargeDeviationInterestingCaseFilter(1.0 - alpha)
compact.runCompleteEvaluation(texTargetFile=latex_file_name ,benchmark=mnistBenchmark,
                              conformanceApproaches=[baseApproach,weightedApproach,],
                              interestingCaseFilter=interestingCaseFilter)

print("\n\n\n")
print("*"*30)
print("Simulation settings:")
print(f"- Seed: {seed_nber}")
print(f"- Dataset: {dataset}")
print(f"- Alpha: {alpha}")
print(f"- Robust training: {is_robust}")
print("- Augmentations:")
for aug, value in dict_augmentations.items():
    print(f"  - {aug}: {value}")
print("*"*30)
print("\n")
print(f"Everything is done and results are written in: {latex_file_name}\n")
