#!/usr/bin/env python3
"""
Entry point for the Rotation-Invariant Ensemble Conformal Prediction evaluation.

This script implements the new ensemble conformal prediction approach:
1. Inputs are characterized by their augmentation-invariance degree d(x), defined as
   the maximum rotation degree d such that rotating the input by any r in [-d, d]
   does not change the model's top-1 predicted class.
2. During precalibration, an optimization problem (MILP) is solved to jointly find the
   optimal bin boundaries [L_b, R_b] and base thresholds tau_b that minimize total
   conformal set size while satisfying the global (1 - alpha) coverage constraint.
3. During calibration, a single multiplicative factor lambda is computed on the calibration
   set to scale the precalibration thresholds: tau_b^{final} = min(1.0, lambda * tau_b),
   achieving the exact finite-sample (1 - alpha) marginal coverage guarantee.
4. During testing, inputs are routed to their respective bin and evaluated, with full
   per-bin breakdown and direct comparison against standard conformal prediction.
"""

import os
import argparse
import numpy as np

import compact
import approaches.rotation_inv_predictor
import approaches.tta_baselines
import interestingcasefilters.largedeviation
from precalibrator.bin_optimizer import compute_test_bin_comparison, format_test_bin_comparison

from utils.function_utils import Augmentation, UtilsAugmentations
from benchmarks.cifar10 import CIFAR10Benchmark
from benchmarks.cifar100 import CIFAR100Benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Rotation-Invariant Ensemble Conformal Prediction")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet", "mnist"])
    parser.add_argument("--alpha", type=float, default=0.1, help="Allowed error level alpha (coverage target is 1 - alpha)")
    parser.add_argument("--n_bins", type=int, default=3, help="Number of invariance bins (default 3)")
    parser.add_argument("--max_rotation", type=int, default=179, help="Maximum rotation angle in degrees")
    parser.add_argument("--min_samples_per_bin", type=int, default=20, help="Minimum samples per bin in precalibration")
    parser.add_argument(
        "--approach", "--routing_metric",
        dest="routing_metric",
        type=str,
        default="symmetric",
        choices=["symmetric", "agreement", "asymmetric_span", "all"],
        help="Ensemble conformal routing approach: "
             "'symmetric' (Continuous symmetric rotation invariance degree d in [0, max_rotation]), "
             "'agreement' (Option 1: Global Agreement Count N_agree in [0, 2*max_rotation]), "
             "'asymmetric_span' (Option 2a: Total Asymmetric Span W = d_L + d_R in [0, 2*max_rotation]), "
             "'all' (Run all three approaches simultaneously for direct side-by-side comparison)."
    )
    parser.add_argument(
        "--scoring_function",
        type=str,
        default=None,
        choices=["thr", "aps"],
        help="Conformal scoring function (thr: 1 - p(y|x), aps: cumulative probability). Defaults to 'aps' for agreement/asymmetric_span and 'thr' for symmetric."
    )
    parser.add_argument("--min_bin_coverage", type=float, default=None, help="Minimum precalibration coverage floor for each bin (e.g. 0.75 or 0.80)")
    parser.add_argument("--coverage_slack", type=float, default=None, help="Anti-cannibalization coverage slack epsilon (e.g. 0.0, 0.02, 0.05). Each bin must satisfy error rate <= alpha + slack with at least 1 error allowed.")
    parser.add_argument("--min_degree_span", type=int, default=2, help="Minimum degree span for candidate bins (prevents width-1 bins)")
    parser.add_argument(
        "--score_in_log_space", "--score-in-log-space",
        dest="score_in_log_space",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute conformal nonconformity scores in log-space (s = -log(p) for THR, -log(1 - s_aps) for APS). "
             "Prevents threshold saturation at 1.0 under extreme coverage targets (e.g. alpha=0.01 / 99%% coverage)."
    )
    parser.add_argument("--robust", action=argparse.BooleanOptionalAction, default=False, help="Use robust trained model if available")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--test_fraction", type=float, default=0.5)
    parser.add_argument("--calibration_fraction", type=float, default=0.4)
    parser.add_argument("--precalibration_fraction", type=float, default=0.1)
    parser.add_argument("--heldout_precal", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    seed_nber = args.seed
    dataset = args.dataset
    alpha = args.alpha
    n_bins = args.n_bins
    max_rotation = args.max_rotation
    is_robust = args.robust
    train_fraction = args.train_fraction
    test_fraction = args.test_fraction
    calibration_fraction = args.calibration_fraction
    precalibration_fraction = args.precalibration_fraction
    heldout_precal = args.heldout_precal
    batch_size = args.batch_size
    min_samples_per_bin = args.min_samples_per_bin
    routing_metric = str(args.routing_metric).lower().strip()
    if args.scoring_function is not None:
        current_scoring_function = args.scoring_function
    else:
        # Default scoring function: aps for Option 1 (agreement) and Option 2a (asymmetric_span), thr for symmetric
        if routing_metric in ["agreement", "asymmetric_span"]:
            current_scoring_function = "aps"
        else:
            current_scoring_function = "thr"

    min_bin_coverage = args.min_bin_coverage
    coverage_slack = args.coverage_slack
    min_degree_span = args.min_degree_span

    TRAIN_ITERATION = 10

    # ImageNet cache contains rotation probes up to 179 degrees
    if dataset == "imagenet" and max_rotation > 179:
        print(f"[imagenet] Note: precomputed cache contains rotation probes up to 179 degrees. Clamping max_rotation from {max_rotation} to 179.")
        max_rotation = 179

    print("\n" + "*" * 70)
    print("RUNNING CONFORMAL PREDICTION EVALUATION")
    print("*" * 70)
    for k, v in vars(args).items():
        print(f"  - {k}: {v}")
    print(f"  -> Selected Approach / Metric: {routing_metric}")
    print(f"  -> Resolved Scoring Function: {current_scoring_function}")
    print("*" * 70 + "\n")

    # Build symmetric rotation probe dictionary: [-max_rotation..-1, 1..max_rotation]
    n_amount_rotation = max_rotation * 2
    list_rotation = UtilsAugmentations.get_n_evenly_spaced_values_for_augmentation(
        augmentation_name=Augmentation.ROTATE,
        lb_degree_aug=-max_rotation,
        ub_degree_aug=max_rotation,
        n_amount=n_amount_rotation,
    )

    dict_augmentations = {}
    for d in list_rotation:
        dict_augmentations[f"ROTATE_PROBE_{d}"] = {"degree": d}

    approach_suffix = "" if routing_metric == "symmetric" else f"_{routing_metric}"
    score_suffix = "" if current_scoring_function == "thr" else f"_{current_scoring_function}"
    log_suffix = "_logspace" if args.score_in_log_space else ""

    # Initialize benchmark
    if dataset in {"cifar10", "cifar100"}:
        benchmark_classes = {
            "cifar10": CIFAR10Benchmark,
            "cifar100": CIFAR100Benchmark,
        }
        benchmark_class = benchmark_classes[dataset]
        latex_file_name = f"Report_{dataset}_{seed_nber}_Robust{is_robust}{approach_suffix}{score_suffix}{log_suffix}.tex"
        path_logs = f"./logs/{dataset}/{seed_nber}/"
        policy_name = "original_only_policy"

        selected_benchmark = benchmark_class(
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
        current_inputs_type = "probs"
        n_leading_views = 1
        leading_views_mode = "individual"
        leading_others_as_aug = True

    elif dataset == "imagenet":
        import benchmarks.imagenet_val
        latex_file_name = f"Report_{dataset}_{seed_nber}_RESNET50{approach_suffix}{score_suffix}{log_suffix}.tex"
        path_logs = f"./logs/imagenetVal/{seed_nber}/"
        current_inputs_type = "logits"
        n_leading_views = 1
        leading_views_mode = "individual"
        leading_others_as_aug = True
        policy_name = "original_only_policy"

        selected_benchmark = benchmarks.imagenet_val.ImageNetValBenchmark(
            policy_name=policy_name,
            seed=seed_nber,
            path_logs=path_logs,
            list_other_aug=dict_augmentations,
            test_fraction=test_fraction,
            calibration_fraction=calibration_fraction,
            precalibration_fraction=precalibration_fraction,
            arch="resnet50",
            weights="IMAGENET1K_V1",
            batch_size=batch_size,
            expected_max_rotation=max_rotation,
        )
        n_classes = selected_benchmark.n_classes
    elif dataset == "mnist":
        import benchmarks.mnist
        latex_file_name = f"Report_{dataset}_{seed_nber}_MNIST{approach_suffix}{score_suffix}.tex"
        path_logs = f"./logs/mnist/{seed_nber}/"
        n_classes = 10
        current_inputs_type = "probs"
        n_leading_views = 1
        leading_views_mode = "individual"
        leading_others_as_aug = True
        selected_benchmark = benchmarks.mnist.MNISTBenchmark(
            path_logs=path_logs,
            seed=seed_nber,
            batch_size=batch_size,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    print(f"\nBenchmark loaded successfully for {dataset} (seed={seed_nber}).")

    # 1. Ensemble Conformal Predictor(s)
    ensemble_predictors = []
    if routing_metric in ["symmetric", "all"]:
        pred_sym = approaches.rotation_inv_predictor.RotationInvariantPredictor(
            precalibration_data=selected_benchmark.precalibration_data,
            alpha=alpha,
            path_logs=path_logs,
            n_classes=n_classes,
            n_bins=n_bins,
            inputs_type=current_inputs_type,
            seed=seed_nber,
            n_leading_views=n_leading_views,
            leading_views_mode=leading_views_mode,
            scoring_function=current_scoring_function if routing_metric != "all" else "thr",
            routing_metric="symmetric",
            verbose=True,
            leading_others_as_aug=leading_others_as_aug,
            max_rotation=max_rotation,
            min_samples_per_bin=min_samples_per_bin,
            min_bin_coverage=min_bin_coverage,
            coverage_slack=coverage_slack,
            min_degree_span=min_degree_span,
            score_in_log_space=args.score_in_log_space,
            symmetric_approach=True,
            n_augs=max_rotation * 2,
        )
        ensemble_predictors.append(pred_sym)

    if routing_metric in ["agreement", "all"]:
        pred_agree = approaches.rotation_inv_predictor.AgreementCountPredictor(
            precalibration_data=selected_benchmark.precalibration_data,
            alpha=alpha,
            path_logs=path_logs,
            n_classes=n_classes,
            n_bins=n_bins,
            inputs_type=current_inputs_type,
            seed=seed_nber,
            n_leading_views=n_leading_views,
            leading_views_mode=leading_views_mode,
            scoring_function=current_scoring_function if routing_metric != "all" else "aps",
            verbose=True,
            leading_others_as_aug=leading_others_as_aug,
            max_rotation=max_rotation,
            min_samples_per_bin=min_samples_per_bin,
            min_bin_coverage=min_bin_coverage,
            coverage_slack=coverage_slack,
            min_degree_span=min_degree_span,
            score_in_log_space=args.score_in_log_space,
            symmetric_approach=True,
            n_augs=max_rotation * 2,
        )
        ensemble_predictors.append(pred_agree)

    if routing_metric in ["asymmetric_span", "all"]:
        pred_span = approaches.rotation_inv_predictor.AsymmetricSpanPredictor(
            precalibration_data=selected_benchmark.precalibration_data,
            alpha=alpha,
            path_logs=path_logs,
            n_classes=n_classes,
            n_bins=n_bins,
            inputs_type=current_inputs_type,
            seed=seed_nber,
            n_leading_views=n_leading_views,
            leading_views_mode=leading_views_mode,
            scoring_function=current_scoring_function if routing_metric != "all" else "aps",
            verbose=True,
            leading_others_as_aug=leading_others_as_aug,
            max_rotation=max_rotation,
            min_samples_per_bin=min_samples_per_bin,
            min_bin_coverage=min_bin_coverage,
            coverage_slack=coverage_slack,
            min_degree_span=min_degree_span,
            score_in_log_space=args.score_in_log_space,
            symmetric_approach=True,
            n_augs=max_rotation * 2,
        )
        ensemble_predictors.append(pred_span)

    # 2. Standard Baseline Conformal Predictor (Single global threshold on original view)
    predictorBaseline = approaches.tta_baselines.OriginalOnlyPredictor(
        precalibration_data=selected_benchmark.precalibration_data,
        alpha=alpha,
        seed=seed_nber,
        n_classes=n_classes,
        path_logs=path_logs,
        scoring_function=current_scoring_function if routing_metric != "all" else "thr",
        inputs_type=current_inputs_type,
        n_leading_views=n_leading_views,
        leading_views_mode=leading_views_mode,
        leading_others_as_aug=leading_others_as_aug,
        score_in_log_space=args.score_in_log_space,
        n_augs=max_rotation * 2,
    )

    conformance_approaches = [*ensemble_predictors, predictorBaseline]

    # Run evaluation via compact framework
    interestingCaseFilter = interestingcasefilters.largedeviation.LargeDeviationInterestingCaseFilter(1.0 - alpha)
    tex_path = os.path.join(path_logs, latex_file_name)

    compact.runCompleteEvaluation(
        texTargetFile=tex_path,
        benchmark=selected_benchmark,
        conformanceApproaches=conformance_approaches,
        interestingCaseFilter=interestingCaseFilter,
        alpha=alpha,
    )

    # Run per-bin comparative evaluation on the test set for each ensemble predictor
    comp_files = []
    for pred in ensemble_predictors:
        test_comp = compute_test_bin_comparison(
            ensemble_predictor=pred,
            baseline_predictor=predictorBaseline,
            test_data=selected_benchmark.testing_data,
            max_rotation=max_rotation,
        )

        comp_report_str = format_test_bin_comparison(test_comp, alpha=alpha)
        print("\n" + comp_report_str + "\n", flush=True)

        # Save comparative test report to disk
        m_tag = f"_{pred.routing_metric}" if pred.routing_metric not in ["symmetric", "sym", "d_sym"] else ""
        s_tag = f"_{pred.scoring_function}" if pred.scoring_function != "thr" else ""
        log_tag = "_logspace" if args.score_in_log_space else ""
        comp_file = os.path.join(path_logs, f"test_bin_comparison{m_tag}_alpha{alpha}_bins{n_bins}{s_tag}{log_tag}.txt")
        with open(comp_file, "w", encoding="utf-8") as f:
            f.write(comp_report_str + "\n")
        comp_files.append(comp_file)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print(f"  - Logs & LaTeX report: {tex_path}")
    for cf in comp_files:
        print(f"  - Test comparison table: {cf}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
