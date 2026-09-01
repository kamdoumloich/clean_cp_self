#!/usr/bin/env python3
# An example for running for COMPACT

import compact
import benchmarks.mnist
import benchmarks.imagenet_val_tta
import benchmarks.imagenet_val
import approaches.base
import approaches.raps
import approaches.tta_baselines
import approaches.aps
import approaches.weightedRankBased
import approaches.weightedRankBasedMeanU
import approaches.weightedRankBasedProbDist
import interestingcasefilters.largedeviation
import benchmarks.resnet

from utils.function_utils import Augmentation
from utils.function_utils import UtilsAugmentations

from benchmarks.cifar10 import CIFAR10Benchmark
from benchmarks.cifar100 import CIFAR100Benchmark

import utils.tta_policies

import argparse
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True, choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
parser.add_argument("--robust", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--min_samples_per_pattern", type=int, default=10)
parser.add_argument("--precalibration_backend", type=str, default="precalibrator",
                    choices=["precalibrator", "solver", "neural"],
)

parser.add_argument("--neural_n_epochs", type=int, default=1000)
parser.add_argument("--neural_lr", type=float, default=1e-3)
parser.add_argument("--neural_weight_decay", type=float, default=1e-4)
parser.add_argument("--neural_temperature_start", type=float, default=0.10)
parser.add_argument("--neural_temperature_end",   type=float, default=0.005)
parser.add_argument("--neural_early_stop_patience", type=int, default=25)
parser.add_argument("--neural_tau_init", type=float, default=0.1,
                    help="Initial tau for 'learnable_tau_pinball'. Pick close to expected operating point.")
parser.add_argument("--neural_tau_lr_multiplier", type=float, default=1.0,
                    help="Multiplier on Adam lr for the tau parameter only.")


parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--test_fraction", type=float, default=0.5)
parser.add_argument("--calibration_fraction", type=float, default=0.4)
parser.add_argument("--precalibration_fraction", type=float, default=0.1)
parser.add_argument("--heldout_precal", action=argparse.BooleanOptionalAction, default=True)

parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

seed_nber = args.seed
dataset = args.dataset
alpha = args.alpha
min_samples_per_pattern = args.min_samples_per_pattern
precalibration_backend = args.precalibration_backend
is_robust = args.robust
train_fraction = args.train_fraction
test_fraction = args.test_fraction
calibration_fraction = args.calibration_fraction
precalibration_fraction = args.precalibration_fraction
heldout_precal=args.heldout_precal
batch_size=args.batch_size

neural_train_kwargs = {
    "n_epochs":             args.neural_n_epochs,
    "batch_size":           args.batch_size,
    "learning_rate":        args.neural_lr,
    "weight_decay":         args.neural_weight_decay,
    "temperature_start":    args.neural_temperature_start,
    "temperature_end":      args.neural_temperature_end,
    "early_stop_patience":  args.neural_early_stop_patience,
    "tau_init":             args.neural_tau_init,
    "tau_lr_multiplier":    args.neural_tau_lr_multiplier,
}

# dict_augmentations = {
#     "ORIGINAL": [0],
#     "ROTATE": [5, 10],
#     # "HORIZONTAL_FLIP": [True],
# }
# current_score = 'raps'
current_scoring_function = 'thr'
max_rotation = 179
# max_rotation = 10
# max_rotation = 5
n_amount_rotation = max_rotation * 2 # without zero

list_rotation_20 = UtilsAugmentations.get_n_evenly_spaced_values_for_augmentation(augmentation_name=Augmentation.ROTATE, lb_degree_aug=-max_rotation, ub_degree_aug=max_rotation, n_amount=n_amount_rotation)


# list_rotation_20 = UtilsAugmentations.get_n_evenly_spaced_values_for_augmentation(augmentation_name=Augmentation.ROTATE, lb_degree_aug=-180, ub_degree_aug=180, n_amount=360)


# list_rotation_20 = UtilsAugmentations.get_n_evenly_spaced_values_for_augmentation(augmentation_name=Augmentation.ROTATE, lb_degree_aug=-10, ub_degree_aug=10, n_amount=20)

# list_rotation_20 = UtilsAugmentations.get_n_evenly_spaced_values_for_augmentation(augmentation_name=Augmentation.ROTATE, lb_degree_aug=-5, ub_degree_aug=5, n_amount=20)

# print(list_rotation_20)
# assert False

dict_augmentations = {
    # "ORIGINAL": [0],
    # "RANDOM_CROP": [4],
    # "HORIZONTAL_FLIP": [True],
    # "ROTATE": list_rotation_21,
}

for d in list_rotation_20:
    # dict_augmentations[f"ROTATE_PROBE_{d}"] = [d]
    dict_augmentations[f"ROTATE_PROBE_{d}"] = {"degree": d}

# dict_tta_simple_policy = utils.tta_policies.build_augs_dict(policy='simple_policy')

TRAIN_ITERATION = 10

print("\n\n")
print("*"*60)
print("RUNNING THE SIMULATION WITH THE FOLLOWING PARAMETERS:")
print("*"*60)
print(' '.join(f' - {k}={v}\n' for k, v in vars(args).items()))
print("*"*60)

latex_file_name = f"ClassWeightedRankBasedRobust{is_robust}.tex"


if dataset in {"cifar10", "cifar100"}:

    benchmark_classes = {
        "cifar10": CIFAR10Benchmark,
        "cifar100": CIFAR100Benchmark,
    }

    benchmark_class = benchmark_classes[dataset]

    latex_file_name = f"Report_{dataset}_{seed_nber}_Robust{is_robust}.tex"
    path_logs = f"./logs/{dataset}/{seed_nber}/"
    
    
    policy_name='original_only_policy'

    selected_benchmark = benchmark_class(
        # augmentations=dict_augmentations,
        policy_name=policy_name,
        list_other_aug=dict_augmentations,
        seed=seed_nber,
        robust_training=is_robust,
        path_logs=path_logs,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        calibration_fraction=calibration_fraction,
        precalibration_fraction=precalibration_fraction,
        iterations=TRAIN_ITERATION,
        heldout_precal=heldout_precal,
        batch_size=batch_size,
        arch="resnet18",
        weights="IMAGENET1K_V1",
        expected_max_rotation=max_rotation,
    )

    n_classes = selected_benchmark.n_classes
    current_inputs_type = 'probs'
    n_leading_views = 1
    leading_views_mode = 'individual'
    leading_others_as_aug = True
    # current_inputs_type = "probabilities"

elif dataset == 'imagenet':

    latex_file_name = f"Report_{dataset}_{seed_nber}_RESNET50.tex"
    path_logs = f"./logs/imagenetVal/{seed_nber}/"
    
    n_classes = 1000
    current_inputs_type = 'logits'
    n_leading_views = 3
    leading_views_mode = 'mean'
    leading_others_as_aug = False
    
    policy_name='simple_policy'
    # policy_name='only_hflip_policy'
    selected_benchmark = benchmarks.imagenet_val_tta.ImageNetValBenchmarkTTA(
        policy_name=policy_name,
        seed=seed_nber,
    #  robust_training=is_robust, 
        path_logs=path_logs,
        list_other_aug=dict_augmentations,
    #  train_fraction = train_fraction,
        test_fraction = test_fraction,
        calibration_fraction = calibration_fraction,
        precalibration_fraction = precalibration_fraction,
    #  iterations=TRAIN_ITERATION,
        arch="resnet50",
        weights="IMAGENET1K_V1",
        batch_size=512,
    )
    


else:
    raise NotImplementedError

print("training for seed: ", seed_nber, " done!")

assert current_inputs_type in ['probs', 'logits'], "Type of inputs must be in ['probs', 'logits'], but {current_inputs_type} given"


# selected_benchmark.evaluate_single_augmentation(
#     augmentation_name=Augmentation.HORIZONTAL_FLIP.value,
#     min_value_augmentation=0,
#     max_value_augmentation=1,
#     n_steps_augmentation=1,
#     dataset_part='training'
# )

#-------------------------------------------------

# from approaches.transformationConsistent import (
#     sanity_check_data, SingleViewScorer, OrbitSymmetricPredictor,
#     TransformationChurnEvaluator)

# SCALE = sanity_check_data(selected_benchmark.calibration_data)["recommended_input_scale"]

# for alpha in (0.1, 0.05, 0.01):
#     for name, sc, bv, tv in [
#         ("Base-THR",  SingleViewScorer("THR", input_scale=SCALE),           [0],   [1]),
#         ("Base-APS",  SingleViewScorer("APS", input_scale=SCALE),           [0],   [1]),
#         ("TC-CP-THR", OrbitSymmetricPredictor([0,1], n_classes, input_scale=SCALE), [0,2], [2,0]),
#     ]:
#         ev = TransformationChurnEvaluator(sc, alpha, bv, tv)
#         ev.calibrate(selected_benchmark.calibration_data)
#         print(alpha, name, ev.evaluate(selected_benchmark.testing_data))
    
# assert False


# #-------------------------------------------------
# # REMARK: Average does not outperform on rotations
# ttaMeanPredictor = approaches.tta_baselines.TTAMeanPredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha, seed=seed_nber,
#     n_classes=n_classes, 
#     # min_samples_per_pattern=min_samples_per_pattern,
#     path_logs = path_logs,
#     scoring_function=current_scoring_function,
#     inputs_type=current_inputs_type,
#     leading_views_mode=leading_views_mode,
#     n_leading_views=n_leading_views,
#     leading_others_as_aug=leading_others_as_aug,
#     n_augs=max_rotation*2,
#     )
# #000000000000000000000000000000000000000000000000000

ttaOrigPredictor = approaches.tta_baselines.OriginalOnlyPredictor(
    precalibration_data=selected_benchmark.precalibration_data,
    alpha=alpha, seed=seed_nber,
    # list_other_aug=dict_augmentations,
    n_classes=n_classes, 
    # min_samples_per_pattern=min_samples_per_pattern,
    path_logs = path_logs,
    scoring_function=current_scoring_function,
    inputs_type=current_inputs_type,
    n_leading_views=n_leading_views,
    leading_views_mode=leading_views_mode,
    leading_others_as_aug=leading_others_as_aug,
    n_augs=max_rotation*2,
    )


# import approaches.transformationConsistent
    
    
    
# tccp = approaches.transformationConsistent.OrbitSymmetricPredictor([0,1], n_classes, input_scale="logits")

# # V, y = self._parse(calibration_data)
# ev = TransformationChurnEvaluator(orbit_indices=[0,1], scorer=tccp, alpha=alpha, base_views=[0, 1], transformed_views=[1, 0], n_classes=n_classes, input_scale="logits")

# ttaGlobalPredictor = approaches.tta_baselines.GlobalWeightPredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha, seed=seed_nber,
#     n_classes=n_classes, 
#     # min_samples_per_pattern=min_samples_per_pattern,
#     path_logs = path_logs,
#     scoring_function=current_scoring_function,
#     inputs_type=current_inputs_type,
#     n_leading_views=n_leading_views,
#     leading_views_mode=leading_views_mode,
#     leading_others_as_aug=leading_others_as_aug,
#     n_augs=max_rotation*2,
# )

# ttaRobustPredictor2 = CandidateAugmentationRankDegradationPredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha,
#     inputs_type=current_inputs_type,
#     path_logs = path_logs,
#     n_classes=n_classes, 
#     n_aggregate=3,
#     probe_indices=tuple(range(3, 24)),
#     gamma_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
#     # gamma_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
#     cv_folds=5,
#     one_standard_error=True,
#     required_relative_gain=0.01,
#     tune_raps=True,
#     seed=seed_nber,
#     verbose=True,
# )

# from approaches.tta_robustSymmetric import SymmetricRotationRankAgreementPredictor

# # rotation_angles = [
# #     0,
# #     *range(-180, 0),
# #     *range(1, 180),
# # ]

# predictor26 = SymmetricRotationRankAgreementPredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha,
#     inputs_type=current_inputs_type,
#     path_logs=path_logs,
#     n_classes=n_classes,
#     score=current_score,
#     n_leading_views=3,
#     leading_views_mode="mean",
#     max_rotation=180,
# )

# from approaches.Rankstabilityweighted import RankStabilityWeightedPredictor

# # rotation_angles = [
# #     0,
# #     *range(-180, 0),
# #     *range(1, 180),
# # ]

# predictor26 = RankStabilityWeightedPredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha,
#     inputs_type=current_inputs_type,
#     path_logs=path_logs,
#     n_classes=n_classes,
#     score=current_score,
#     n_leading_views=3,
#     leading_views_mode="mean",
#         # weight_mode="stable_count",
#         weight_mode="scaled",
#     max_rotation=10,
# )


# import approaches.quantile_cell_thr_predictor

# predictorNone = approaches.quantile_cell_thr_predictor.QuantileCellTHRPredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha,
#     path_logs=path_logs,
#     n_classes=n_classes,
#     inputs_type="logits",   # use "probs" if cache already contains probabilities
#     n_bins_1=6,
#     n_bins_2=3,
#     n_bins_3=3,
#     # n_bins_4=8,
#     seed=seed_nber,
#     n_leading_views=3,
#     leading_views_mode="mean",
#     score=current_score,
#     verbose=True,
#     max_rotation=max_rotation,
# )

import approaches.rotation_inv_predictor

predictorSym = approaches.rotation_inv_predictor.RotationInvariantPredictor(
    precalibration_data=selected_benchmark.precalibration_data,
    alpha=alpha,
    path_logs=path_logs,
    n_classes=n_classes,
    inputs_type=current_inputs_type,   # use "probs" if cache already contains probabilities
    seed=seed_nber,
    n_leading_views=n_leading_views,
    leading_views_mode=leading_views_mode,
    scoring_function=current_scoring_function,
    verbose=True,
    leading_others_as_aug=leading_others_as_aug,
    max_rotation=max_rotation,
    symmetric_approach = True,
    n_augs=max_rotation*2,
)

# predictorAntiSym = approaches.rotation_inv_predictor.RotationInvariantPredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha,
#     path_logs=path_logs,
#     n_classes=n_classes,
#     inputs_type=current_inputs_type,   # use "probs" if cache already contains probabilities
#     seed=seed_nber,
#     n_leading_views=n_leading_views,
#     leading_views_mode=leading_views_mode,
#     score=current_score,
#     verbose=True,
#     leading_others_as_aug=leading_others_as_aug,
#     max_rotation=max_rotation,
#     symmetric_approach = False,
# )



# conformance_approaches = [ttaRobustPredictor, gateApproach, ttaMeanPredictor, baseApproach]
# conformance_approaches = [ttaStabilityPredictor,]
# conformance_approaches = [ttaStabilityPredictor, ttaMeanPredictor,]
# conformance_approaches = [predictorSym]
conformance_approaches = [predictorSym, ttaOrigPredictor,]
# conformance_approaches = [predictorNone, ttaOrigPredictor, ttaMeanPredictor, ttaGlobalPredictor]
# conformance_approaches = [predictor26, rapsApproach, ttaGlobalPredictor, ]

import churnfilter

# assert False
interestingCaseFilter = interestingcasefilters.largedeviation.LargeDeviationInterestingCaseFilter(1.0 - alpha)
# interestingCaseFilter = churnfilter.ChurnInterestingCaseFilter(
#     1.0 - alpha,
#     predictors=conformance_approaches,   # same list, same order
#     sigma=(1, 0),                        # [ORIGINAL, HORIZONTAL_FLIP]
#     n_classes=n_classes,
#     anchorName="HORIZONTAL_FLIP",
# )
compact.runCompleteEvaluation(texTargetFile=str(path_logs)+""+str(latex_file_name), benchmark=selected_benchmark,
                              conformanceApproaches=conformance_approaches,
                              interestingCaseFilter=interestingCaseFilter,
                              alpha=alpha,)




print("\n\n\n")
print("*"*30)
print("SIMULATION SETTINGS:")
for name, value in vars(args).items():
    print(f"  - {name}: {value}")
# print("- AUGMENTATIONS:")
# for aug, value in dict_augmentations.items():
#     print(f"  - {aug}: {value}")
print("*"*30)
print("\n")
print(f"Everything is done and results are written in: {path_logs}\n")



assert False


# weightedApproachExact = approaches.weightedRankBased.WeightedRank2DPredictor(precalibration_data=selected_benchmark.precalibration_data,
#                                                                         alpha=alpha, seed=seed_nber,
#                                                                         n_classes=n_classes, min_samples_per_pattern=min_samples_per_pattern,
#                                                                         path_logs = path_logs,
#                                                                         precalibration_backend="precalibrator",
#                                                                         neural_train_kwargs=neural_train_kwargs,
#                                                                         )

# weightedApproachNN = approaches.weightedRankBased.WeightedRank2DPredictor(precalibration_data=selected_benchmark.precalibration_data,
#                                                                         alpha=alpha, seed=seed_nber,
#                                                                         n_classes=n_classes, min_samples_per_pattern=min_samples_per_pattern,
#                                                                         path_logs = path_logs,
#                                                                         precalibration_backend="neural",
#                                                                         neural_train_kwargs=neural_train_kwargs,
#                                                                         )


# weightedApproachMeanU = approaches.weightedRankBasedMeanU.WeightedRankMeanU2DPredictor(precalibration_data=selected_benchmark.precalibration_data,
#                                                                         alpha=alpha, seed=seed_nber,
#                                                                         n_classes=n_classes, min_samples_per_pattern=min_samples_per_pattern,
#                                                                         path_logs = path_logs,
#                                                                         precalibration_backend=precalibration_backend,
#                                                                         neural_train_kwargs=neural_train_kwargs,
#                                                                         )

# weightedApproachProbDist = approaches.weightedRankBasedProbDist.WeightedRankProbDist2DPredictor(precalibration_data=selected_benchmark.precalibration_data,
#                                                                         alpha=alpha, seed=seed_nber,
#                                                                         n_classes=n_classes, min_samples_per_pattern=min_samples_per_pattern,
#                                                                         path_logs = path_logs,
#                                                                         precalibration_backend="neural",
#                                                                         neural_train_kwargs=neural_train_kwargs,
#                                                                         )

baseApproach = approaches.base.MonotoneConformanceEvaluatorSingleDistributionAllAboveThresholdButAtLeastOneClass()
# conformance_approaches = [baseApproach, weightedApproach]

# import approaches.quantileCellMixture
# qcmApproach = approaches.quantileCellMixture.QuantileCellMixturePredictor(
#     precalibration_data=selected_benchmark.precalibration_data,
#     alpha=alpha, seed=seed_nber, n_classes=n_classes, path_logs=path_logs,)


import approaches.quantileCellMixture_distance
qcmApproach_distance = approaches.quantileCellMixture_distance.QuantileCellMixturePredictor(
    precalibration_data=selected_benchmark.precalibration_data,
    alpha=alpha, seed=seed_nber, n_classes=n_classes, path_logs=path_logs,)




# conformance_approaches = [baseApproach, rapsApproach, weightedApproach]
# conformance_approaches = [baseApproach, rapsApproach, weightedApproach]
# conformance_approaches = [baseApproach, qcmApproach, qcmApproach_distance, gateApproach, gateApproach_milp]
conformance_approaches = [baseApproach, gateApproach_milp, qcmApproach_distance, gateApproach_milp_o]
# conformance_approaches = [gateApproach, gateApproach_aps, gateApproach_milp, qcmApproach_distance]
# conformance_approaches = [qcmApproach_distance, gateApproach_milp]
# conformance_approaches = [baseApproach, gateApproach, weightedApproachNN]
# conformance_approaches = [baseApproach, gateApproach,]


# assert False
interestingCaseFilter = interestingcasefilters.largedeviation.LargeDeviationInterestingCaseFilter(1.0 - alpha)
compact.runCompleteEvaluation(texTargetFile=str(path_logs)+""+str(latex_file_name), benchmark=selected_benchmark,
                              conformanceApproaches=conformance_approaches,
                              interestingCaseFilter=interestingCaseFilter,
                              alpha=alpha,)

print("\n\n\n")
print("*"*30)
print("SIMULATION SETTINGS:")
for name, value in vars(args).items():
    print(f"  - {name}: {value}")
print("- AUGMENTATIONS:")
for aug, value in dict_augmentations.items():
    print(f"  - {aug}: {value}")
print("*"*30)
print("\n")
print(f"Everything is done and results are written in: {path_logs}\n")

# python3 runWeightedRankBased.py --seed 2026 --dataset cifar100 --alpha 0.1 --train_fraction 1.0 --test_fraction 0.8 --precalibration_fraction 1.0 --min_samples_per_pattern 10 --precalibration_backend neural --neural_tau_init 0.1 --neural_n_epochs 700
# python3 runWeightedRankBased.py --seed 2026 --dataset cifar10 --alpha 0.1 --train_fraction 1.0 --test_fraction 0.8 --precalibration_fraction 0.2 --min_samples_per_pattern 10 --precalibration_backend neural --neural_n_epochs 600 --neural_batch_size 128
