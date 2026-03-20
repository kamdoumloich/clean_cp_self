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
parser.add_argument("--robust", type=bool, choices=['True', 'False'], default=False)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--min_samples_per_pattern", type=int, default=10)

# TODO: Minimal settings for my experiments
parser.add_argument("--train_fraction", type=float, default=0.1)
parser.add_argument("--test_fraction", type=float, default=0.1)
parser.add_argument("--calibration_fraction", type=float, default=0.2)
parser.add_argument("--precalibration_fraction", type=float, default=0.1)

args = parser.parse_args()

seed_nber = args.seed
dataset = args.dataset
alpha = args.alpha
min_samples_per_pattern = args.min_samples_per_pattern
is_robust = args.robust
train_fraction = args.train_fraction
test_fraction = args.test_fraction
calibration_fraction = args.calibration_fraction
precalibration_fraction = args.precalibration_fraction

dict_augmentations = {
    "ORIGINAL": 0,
    # "ROTATION": 3,
    # "HUE": 0.01,
    "CONTRAST": 1.2,
    # "HORIZONTAL_FLIP": True,
}

TRAIN_ITERATION = 4

print("\n\n")
print("*"*60)
print("RUNNING THE SIMULATION WITH THE FOLLOWING PARAMETERS:")
print("*"*60)
print(' '.join(f' - {k}={v}\n' for k, v in vars(args).items()))
print("*"*60)

latex_file_name = f"ClassWeightedRankBasedRobust{is_robust}.tex"


if dataset == 'cifar':

    latex_file_name = f"ClassWeightedRankBased{seed_nber}Robust{is_robust}.tex"

    path_logs = "./logs/cifar/"+str(seed_nber)+"/"

    selected_benchmark = benchmarks.cifar.CIFAR10Benchmark(augmentations=dict_augmentations, seed=seed_nber,
                                                         robust_training=is_robust, path_logs=path_logs,
                                                         train_fraction = train_fraction,
                                                         test_fraction = test_fraction,
                                                         calibration_fraction = calibration_fraction,
                                                         precalibration_fraction = precalibration_fraction,
                                                           iterations=TRAIN_ITERATION,
                                                        )
    n_classes = 10
elif dataset == 'mnist':

    raise NotImplementedError("Not implemented yet.")

    path_logs = "./logs/mnist/"+str(seed_nber)+"/"

    selected_benchmark = benchmarks.mnist.MNISTBenchmark(augmentations=dict_augmentations, seed=seed_nber,
                                                         robust_training=is_robust, path_logs=path_logs)
    n_classes = 10

else:
    raise NotImplementedError

baseApproach = approaches.base.MonotoneConformanceEvaluatorSingleDistributionAllAboveThresholdButAtLeastOneClass()
weightedApproach = approaches.weightedRankBased.WeightedRank2DPredictor(precalibration_data=selected_benchmark.precalibration_data,
                                                                        alpha=alpha, seed=seed_nber,
                                                                        n_classes=n_classes, min_samples_per_pattern=min_samples_per_pattern,
                                                                        path_logs = path_logs,
                                                                        )
# assert False
interestingCaseFilter = interestingcasefilters.largedeviation.LargeDeviationInterestingCaseFilter(1.0 - alpha)
compact.runCompleteEvaluation(texTargetFile=str(path_logs)+""+str(latex_file_name), benchmark=selected_benchmark,
                              conformanceApproaches=[baseApproach,weightedApproach,],
                              interestingCaseFilter=interestingCaseFilter)

print("\n\n\n")
print("*"*30)
print("SIMULATION SETTINGS:")
for name, value in parser.parse_args().__dict__.items():
    print(f"  - {name}: {value}")
print("- AUGMENTATIONS:")
for aug, value in dict_augmentations.items():
    print(f"  - {aug}: {value}")
print("*"*30)
print("\n")
print(f"Everything is done and results are written in: {path_logs}\n")
