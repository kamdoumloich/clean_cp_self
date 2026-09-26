"""
Evaluation metrics for conformal prediction.

All metrics are computed from per-test-sample records:
    size_i      : |C(x_i)|
    covered_i   : 1[y_i in C(x_i)]
    y_i         : true class index
    difficulty_i: input-side difficulty, e.g. 1 - max_k p_k(x_i)
"""

import numpy as np


def _coverage_std_error(coverage, n):
    if n <= 0:
        return float("nan")

    return float(
        np.sqrt(
            max(coverage * (1.0 - coverage), 0.0) / n
        )
    )


def compute_CovGap(covered, difficulty, target, n_bins=10, ):
    """
    Coverage Gap over input-side difficulty strata.

    CovGap = mean_b | coverage_b - target |

    where:
        target = 1 - alpha

    difficulty should be predictor-independent, for example:

        difficulty(x) = 1 - max_k p_k(x)

    Lower is better.

    Returns
    -------
    {
        "CovGap"          : mean absolute coverage gap,
        "CovGap_pct"      : same quantity in percent,
        "CovGap_weighted" : sample-count-weighted CovGap,
        "worst_under"     : most negative coverage deviation,
        "bins"            : [(lo, hi, count, coverage), ...]
    }
    """

    covered = np.asarray(covered, dtype=np.float64,)
    difficulty = np.asarray(difficulty, dtype=np.float64,)

    if covered.ndim != 1:
        raise ValueError("covered must be one-dimensional.")

    if difficulty.ndim != 1:
        raise ValueError("difficulty must be one-dimensional.")

    if len(covered) != len(difficulty):
        raise ValueError("covered and difficulty must have the same length.")

    n = len(covered)

    if n == 0:
        return None

    if not np.all(np.isfinite(covered)):
        raise ValueError("covered contains non-finite values.")

    if not np.all(np.isfinite(difficulty)):
        raise ValueError("difficulty contains non-finite values.")

    if np.any((covered < 0.0) | (covered > 1.0)):
        raise ValueError("covered must contain only values in [0,1].")

    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0,1].")

    # This catches the previous logits/probabilities bug immediately.
    if np.any((difficulty < 0.0) | (difficulty > 1.0)):
        raise ValueError(
            "difficulty must lie in [0,1]. If difficulty = 1 - max(p_orig), make sure p_orig "
            "contains probabilities rather than logits."
        )

    # ---------------------------------------------------------
    # Quantile bins of the input-side difficulty variable
    # ---------------------------------------------------------
    edges = np.quantile(difficulty, np.linspace(0.0, 1.0, n_bins + 1,),)

    # Remove duplicate interior edges.
    # This can happen when many samples have identical difficulty.
    edges = np.unique(edges)

    if len(edges) < 2:
        # All difficulty values are identical.
        coverage = float(covered.mean())
        gap = abs(coverage - target)

        return {
            "CovGap": float(gap),
            "CovGap_pct": float(100.0 * gap),
            "CovGap_weighted": float(gap),
            "worst_under": float(min(0.0, coverage - target)),
            "bins": [(float(difficulty[0]), float(difficulty[0]), n, coverage,)],
        }

    # Include maximum in final interval.
    edges[-1] = np.nextafter(edges[-1], np.inf,)

    bin_id = np.digitize(difficulty, edges[1:-1], right=False,)

    gaps = []
    counts = []
    undercoverage = []
    rows = []

    n_effective_bins = len(edges) - 1

    for b in range(n_effective_bins):

        mask = bin_id == b
        count = int(mask.sum())

        if count == 0:
            continue

        coverage_b = float(covered[mask].mean())
        deviation_b = (coverage_b - target)

        gaps.append(abs(deviation_b))
        counts.append(count)
        undercoverage.append(deviation_b)
        rows.append((float(edges[b]), float(edges[b + 1]), count, coverage_b,))

    if not gaps:
        return None

    gaps = np.asarray(gaps, dtype=np.float64,)
    counts = np.asarray(counts, dtype=np.float64,)

    # Equal-stratum CovGap
    covgap = float(gaps.mean())

    # Sample-count-weighted variant
    covgap_weighted = float(np.average(gaps, weights=counts,))
    worst_under = float(min(0.0, min(undercoverage),))

    return {
        "CovGap": covgap,
        "CovGap_pct": 100.0 * covgap,
        "CovGap_weighted": covgap_weighted,
        "worst_under": worst_under,
        "bins": rows,
    }


def compute_all_metrics(sizes, covered, y_true, alpha, n_classes, size_bins=None, large_threshold=None, difficulty=None, ):
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    sizes : array-like
        Prediction-set sizes.

    covered : array-like
        1 if true label belongs to prediction set, otherwise 0.

    y_true : array-like
        True class indices.

    alpha : float
        Target error rate.

    n_classes : int
        Number of classes.

    difficulty : optional array-like
        Predictor-independent input difficulty.
        Recommended:
            1 - max_k p_orig[k]
    """

    sizes = np.asarray(sizes, dtype=int,)
    covered = np.asarray(covered, dtype=float,)
    y_true = np.asarray(y_true, dtype=int,)

    n = sizes.shape[0]
    target = 1.0 - alpha

    if n == 0:
        return {"n": 0}

    if len(covered) != n or len(y_true) != n:
        raise ValueError("sizes, covered and y_true must have equal length.")

    if large_threshold is None:
        large_threshold = max(3, int(np.ceil(n_classes / 4)),)

    # =========================================================
    # 1. Size-Stratified Coverage Violation
    # =========================================================
    if size_bins is None:
        edges = [1, 3, 6, 10,]
    else:
        edges = list(size_bins)

    ranges = []
    lo = 1

    for edge in edges:

        if edge < lo:
            continue

        ranges.append((lo, edge,))
        lo = edge + 1

    if lo <= n_classes:
        ranges.append((lo, n_classes,))

    sscv_rows = []
    worst = 0.0

    for rlo, rhi in ranges:

        mask = ((sizes >= rlo) & (sizes <= rhi))
        count = int(mask.sum())

        if count == 0:

            sscv_rows.append((rlo, rhi, 0, float("nan"),))

            continue

        coverage_bin = float(covered[mask].mean())

        sscv_rows.append((rlo, rhi, count, coverage_bin,))

        worst = max(worst, abs(coverage_bin - target),)

    sscv = float(worst)

    # =========================================================
    # 2. Class-stratified coverage
    # =========================================================
    class_cov = []

    for k in range(
        n_classes
    ):

        mask = (
            y_true == k
        )

        count = int(
            mask.sum()
        )

        if count == 0:

            class_cov.append(
                (
                    k,
                    0,
                    float("nan"),
                )
            )

        else:

            class_cov.append(
                (
                    k,
                    count,
                    float(
                        covered[
                            mask
                        ].mean()
                    ),
                )
            )

    valid = np.asarray(
        [
            coverage_k
            for (
                _,
                count,
                coverage_k,
            ) in class_cov
            if count > 0
        ],
        dtype=float,
    )

    if valid.size:

        class_cov_min = float(
            valid.min()
        )

        class_cov_max = float(
            valid.max()
        )

        class_cov_spread = (
            class_cov_max
            - class_cov_min
        )

        class_cov_std = float(
            valid.std()
        )

    else:

        class_cov_min = float(
            "nan"
        )

        class_cov_max = float(
            "nan"
        )

        class_cov_spread = float(
            "nan"
        )

        class_cov_std = float(
            "nan"
        )

    # =========================================================
    # 3. Set-size distribution
    # =========================================================
    median_size = float(
        np.median(
            sizes
        )
    )

    singleton_rate = float(
        np.mean(
            sizes == 1
        )
    )

    large_rate = float(
        np.mean(
            sizes
            >= large_threshold
        )
    )

    empty_rate = float(
        np.mean(
            sizes == 0
        )
    )

    # =========================================================
    # 4. Marginal coverage
    # =========================================================
    coverage = float(
        covered.mean()
    )

    coverage_se = (
        _coverage_std_error(
            coverage,
            n,
        )
    )

    # =========================================================
    # 5. CovGap
    # =========================================================
    covgap = None

    if difficulty is not None:

        covgap = compute_CovGap(
            covered=covered,
            difficulty=difficulty,
            target=target,
        )

    return {
        "n": n,
        "alpha": alpha,
        "target_coverage": target,

        "coverage": coverage,
        "coverage_se": coverage_se,

        "avg_size": float(sizes.mean()),

        "median_size": median_size,
        "singleton_rate": singleton_rate,
        "large_rate": large_rate,
        "large_threshold": large_threshold,
        "empty_rate": empty_rate,

        "sscv": sscv,
        "sscv_rows": sscv_rows,

        "class_cov": class_cov,
        "class_cov_min": class_cov_min,
        "class_cov_max": class_cov_max,
        "class_cov_spread": class_cov_spread,
        "class_cov_std": class_cov_std,

        "CovGap": covgap,
    }


def _fmt(
    x,
    nd=4,
):
    if x is None:
        return "--"

    if (
        isinstance(
            x,
            float,
        )
        and np.isnan(
            x
        )
    ):
        return "--"

    return (
        f"{x:.{nd}f}"
    )


def metrics_to_tex(
    metrics,
    short_name,
):
    """
    Compact LaTeX report for one predictor.
    """

    if metrics.get("n", 0,) == 0:
        return (
            f"\\paragraph{{{short_name}}} "
            "No test cases.\\\\\n"
        )

    out = []

    out.append(
        f"\\paragraph{{Metrics for {short_name}}}"
        "~\\\\\n"
    )

    # =========================================================
    # Summary table
    # =========================================================
    out.append(
        "\\begin{tabular}{l r}\n"
        "\\hline\n"
    )

    out.append(
        f"Coverage "
        f"(target {_fmt(metrics['target_coverage'], 3)}) "
        f"& {_fmt(metrics['coverage'])} "
        f"$\\pm$ {_fmt(metrics['coverage_se'])} "
        "\\\\\n"
    )

    out.append(
        f"Average set size "
        f"& {_fmt(metrics['avg_size'], 3)} "
        "\\\\\n"
    )

    out.append(
        f"Median set size "
        f"& {_fmt(metrics['median_size'], 1)} "
        "\\\\\n"
    )

    out.append(
        f"Singleton rate (size $=1$) "
        f"& {_fmt(metrics['singleton_rate'])} "
        "\\\\\n"
    )

    out.append(
        f"Large-set rate "
        f"(size $\\geq "
        f"{metrics['large_threshold']}$) "
        f"& {_fmt(metrics['large_rate'])} "
        "\\\\\n"
    )

    if (
        metrics[
            "empty_rate"
        ]
        > 0.0
    ):

        out.append(
            f"Empty-set rate "
            f"(size $=0$) "
            f"& {_fmt(metrics['empty_rate'])} "
            "\\\\\n"
        )

    out.append(
        f"\\textbf{{SSCV}} "
        f"(worst size-bin gap) "
        f"& \\textbf{{{_fmt(metrics['sscv'])}}} "
        "\\\\\n"
    )

    out.append(
        f"Class-coverage spread "
        f"(max$-$min) "
        f"& {_fmt(metrics['class_cov_spread'])} "
        "\\\\\n"
    )

    out.append(
        f"Class-coverage std "
        f"& {_fmt(metrics['class_cov_std'])} "
        "\\\\\n"
    )

    covgap = metrics.get("CovGap")

    if covgap is not None:

        out.append(
            f"\\textbf{{CovGap}} "
            f"(difficulty-conditioned) "
            f"& \\textbf{{{_fmt(covgap['CovGap'])}}} "
            "\\\\\n"
        )

        out.append(
            f"~~Weighted CovGap "
            f"& {_fmt(covgap['CovGap_weighted'])} "
            "\\\\\n"
        )

        out.append(
            f"~~Worst under-coverage "
            f"& {_fmt(covgap['worst_under'])} "
            "\\\\\n"
        )

    out.append(
        "\\hline\n"
        "\\end{tabular}"
        "\\\\[0.5em]\n"
    )

    # =========================================================
    # SSCV table
    # =========================================================
    out.append(
        "\\textsf{\\scriptsize "
        "Size-stratified coverage "
        "(SSCV bins):}"
        "\\\\\n"
    )

    out.append(
        "\\begin{tabular}{l r r}\n"
        "\\hline\n"
    )

    out.append(
        "Size range & Count & Coverage "
        "\\\\\n"
        "\\hline\n"
    )

    for (
        rlo,
        rhi,
        count,
        coverage_bin,
    ) in metrics[
        "sscv_rows"
    ]:

        if rlo == rhi:
            label = (
                f"{rlo}"
            )
        else:
            label = (
                f"{rlo}--{rhi}"
            )

        out.append(
            f"{label} "
            f"& {count} "
            f"& {_fmt(coverage_bin)} "
            "\\\\\n"
        )

    out.append(
        "\\hline\n"
        "\\end{tabular}"
        "\\\\[0.5em]\n"
    )

    # =========================================================
    # CovGap table
    # =========================================================
    if covgap is not None:

        out.append(
            "\\textsf{\\scriptsize "
            "Difficulty-stratified coverage "
            "(CovGap bins, by "
            "$1-\\max p_{\\mathrm{orig}}$):}"
            "\\\\\n"
        )

        out.append(
            "\\begin{tabular}{l r r}\n"
            "\\hline\n"
        )

        out.append(
            "Difficulty range "
            "& Count "
            "& Coverage "
            "\\\\\n"
            "\\hline\n"
        )

        for (lo, hi, count, coverage_bin,) in covgap["bins"]:

            out.append(
                f"{_fmt(lo, 3)}--{_fmt(hi, 3)} "
                f"& {count} "
                f"& {_fmt(coverage_bin)} "
                "\\\\\n"
            )

        out.append(
            "\\hline\n"
            "\\end{tabular}"
            "\\\\[0.5em]\n"
        )

    return "".join(
        out
    )


def metrics_comparison_table(
    per_predictor,
):
    """
    One row per predictor.
    """

    out = []

    out.append(
        "\\begin{tabular}"
        "{l r r r r r r r r}\n"
        "\\hline\n"
    )

    out.append(
        "Predictor "
        "& Coverage "
        "& Avg size "
        "& Median "
        "& Singleton "
        "& Large "
        "& SSCV "
        "& CovGap "
        "& wUnder "
        "\\\\\n"
        "\\hline\n"
    )

    for (
        short_name,
        metrics,
    ) in per_predictor:

        if metrics.get(
            "n",
            0,
        ) == 0:

            out.append(
                f"{short_name} "
                "& \\multicolumn{8}{c}"
                "{no test cases} "
                "\\\\\n"
            )

            continue

        covgap = metrics.get(
            "CovGap"
        )

        if covgap is None:

            covgap_value = "--"
            worst_under = "--"

        else:

            covgap_value = _fmt(
                covgap[
                    "CovGap"
                ]
            )

            worst_under = _fmt(
                covgap[
                    "worst_under"
                ]
            )

        out.append(
            f"{short_name} "
            f"& {_fmt(metrics['coverage'])} "
            f"& {_fmt(metrics['avg_size'], 3)} "
            f"& {_fmt(metrics['median_size'], 1)} "
            f"& {_fmt(metrics['singleton_rate'])} "
            f"& {_fmt(metrics['large_rate'])} "
            f"& {_fmt(metrics['sscv'])} "
            f"& {covgap_value} "
            f"& {worst_under} "
            "\\\\\n"
        )

    out.append(
        "\\hline\n"
        "\\end{tabular}"
        "\\\\[0.5em]\n"
    )

    out.append(
        "\\textsf{\\scriptsize "
        "SSCV = worst absolute coverage gap "
        "across prediction-set-size strata. "
        "CovGap = mean absolute deviation of "
        "difficulty-stratified coverage from "
        "the target $1-\\alpha$. "
        "wUnder = worst difficulty-stratum "
        "under-coverage; values closer to zero "
        "are better.}"
        "\\\\\n"
    )

    return "".join(
        out
    )