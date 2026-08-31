"""
statistical_tests_mitbih.py

Runs a full battery of statistical significance tests comparing your four
MIT-BIH models (BaselineCNN, ResCNN, LiteECGCNN, LiteECGDSCNN), following
international-standard practice for comparing deep learning classifiers:

1. McNemar's test (pairwise, single-run comparison) + phi effect size
   + Bonferroni-corrected significance threshold.
2. Per-class Chi-square / Fisher's exact test (N, S, V, F, Q) for each
   pairwise model comparison.
3. Bootstrap 95% confidence intervals for accuracy and macro-F1 for
   each model individually.

Requirements (install once in your ecg-thesis conda env):
    pip install statsmodels scipy scikit-learn pandas numpy

Expected input files (already produced by your training scripts), all in
the same RESULTS_DIR:
    mitbih_all_baseline_predictions.csv
    mitbih_all_rescnn_predictions.csv
    mitbih_all_liteecgcnn_predictions.csv
    mitbih_all_liteecgdscnn_predictions.csv

Each file must have columns: index, y_true, y_pred

Output:
    results/statistical_tests_summary.xlsx
        - Sheet 'mcnemar_pairwise'      : McNemar chi2, p-value, phi, significance
        - Sheet 'per_class_chi2_fisher' : per-class contingency test results
        - Sheet 'bootstrap_ci'          : bootstrap CIs for accuracy & macro-F1
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = os.getcwd()
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_FILES = {
    "BaselineCNN":   "mitbih_all_baseline_predictions.csv",
    "ResCNN":        "mitbih_all_rescnn_predictions.csv",
    "LiteECGCNN":    "mitbih_all_liteecgcnn_predictions.csv",
    "LiteECGDSCNN":  "mitbih_all_liteecgdscnn_predictions.csv",
}

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
ALPHA = 0.05  # base significance level, before Bonferroni correction

np.random.seed(RANDOM_SEED)


# =============================================================================
# Loading and alignment helpers
# =============================================================================

def load_predictions(results_dir, model_files):
    """
    Load each model's predictions CSV. Returns dict:
        {model_name: DataFrame[index, y_true, y_pred]}
    """
    data = {}
    for model_name, filename in model_files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] Missing file for {model_name}: {path} -- skipping.")
            continue
        df = pd.read_csv(path)
        required_cols = {"index", "y_true", "y_pred"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"{filename} must contain columns {required_cols}, "
                f"found {list(df.columns)}"
            )
        data[model_name] = df
    if len(data) < 2:
        raise RuntimeError(
            "Need at least 2 model prediction files to run comparisons."
        )
    return data


def align_by_index(df_a, df_b):
    """
    Align two prediction DataFrames on their 'index' column so that
    comparisons are made on matching validation samples.

    NOTE: This assumes all models were evaluated on comparable validation
    sets sharing the same 'index' numbering scheme (as produced when each
    training script uses the same random_state=42 stratified split logic).
    If your splits differ between models, this alignment will not be valid;
    in that case you must re-run all models on one shared, saved split.
    """
    merged = pd.merge(
        df_a, df_b, on="index", suffixes=("_a", "_b"), how="inner"
    )
    if merged.empty:
        raise RuntimeError(
            "No overlapping indices between the two models' predictions. "
            "Ensure both models were evaluated on the same validation split."
        )
    # sanity check: true labels must match after alignment
    mismatched = (merged["y_true_a"] != merged["y_true_b"]).sum()
    if mismatched > 0:
        print(
            f"[WARN] {mismatched} rows have mismatched y_true between "
            f"the two models after aligning on 'index'. This suggests the "
            f"two models used DIFFERENT train/val splits, which invalidates "
            f"a paired McNemar comparison. Proceed with caution."
        )
    return merged


# =============================================================================
# 1. McNemar's test + phi effect size
# =============================================================================

def run_mcnemar_pairwise(data):
    """
    For every pair of models, build the 2x2 contingency table of
    correct/incorrect predictions and run McNemar's exact test.

    Table layout:
                    Model B correct   Model B wrong
    Model A correct       n11              n10
    Model A wrong          n01              n00
    """
    model_names = list(data.keys())
    pairs = list(itertools.combinations(model_names, 2))
    n_pairs = len(pairs)
    bonferroni_alpha = ALPHA / n_pairs if n_pairs > 0 else ALPHA

    rows = []
    for model_a, model_b in pairs:
        merged = align_by_index(data[model_a], data[model_b])

        correct_a = (merged["y_pred_a"] == merged["y_true_a"]).astype(int)
        correct_b = (merged["y_pred_b"] == merged["y_true_b"]).astype(int)

        n11 = int(((correct_a == 1) & (correct_b == 1)).sum())
        n10 = int(((correct_a == 1) & (correct_b == 0)).sum())
        n01 = int(((correct_a == 0) & (correct_b == 1)).sum())
        n00 = int(((correct_a == 0) & (correct_b == 0)).sum())

        table = np.array([[n11, n10], [n01, n00]])

        # McNemar's test (exact=True recommended when b+c is small,
        # otherwise chi-square approximation with continuity correction)
        b_plus_c = n10 + n01
        use_exact = b_plus_c < 25
        result = mcnemar(table, exact=use_exact, correction=not use_exact)

        # Phi effect size for McNemar (based on discordant pairs)
        if b_plus_c > 0:
            phi = abs(n10 - n01) / np.sqrt(b_plus_c)
        else:
            phi = 0.0

        acc_a = correct_a.mean()
        acc_b = correct_b.mean()

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "acc_a": round(acc_a, 4),
            "acc_b": round(acc_b, 4),
            "n_both_correct": n11,
            "n_a_only_correct": n10,
            "n_b_only_correct": n01,
            "n_both_wrong": n00,
            "mcnemar_statistic": round(float(result.statistic), 4),
            "p_value": result.pvalue,
            "phi_effect_size": round(phi, 4),
            "bonferroni_alpha": round(bonferroni_alpha, 5),
            "significant_after_bonferroni": bool(result.pvalue < bonferroni_alpha),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 2. Per-class Chi-square / Fisher's exact test
# =============================================================================

def run_per_class_tests(data):
    """
    For each pair of models and each AAMI class, test whether the
    per-class correctness distribution differs significantly.

    Contingency table per class c, per model:
                  Correct on class c   Incorrect on class c
    (counts among samples where y_true == c)

    We compare model_a's per-class correctness vector against model_b's.
    """
    model_names = list(data.keys())
    pairs = list(itertools.combinations(model_names, 2))

    rows = []
    for model_a, model_b in pairs:
        merged = align_by_index(data[model_a], data[model_b])

        for class_id, class_name in CLASS_ID_TO_NAME.items():
            subset = merged[merged["y_true_a"] == class_id]
            n_samples = len(subset)
            if n_samples == 0:
                continue

            correct_a = (subset["y_pred_a"] == subset["y_true_a"]).astype(int)
            correct_b = (subset["y_pred_b"] == subset["y_true_b"]).astype(int)

            a_correct = int(correct_a.sum())
            a_wrong = n_samples - a_correct
            b_correct = int(correct_b.sum())
            b_wrong = n_samples - b_correct

            table = np.array([[a_correct, a_wrong], [b_correct, b_wrong]])

            # Use Fisher's exact test when any expected cell count is small,
            # otherwise chi-square test of independence.
            row_sums = table.sum(axis=1)
            col_sums = table.sum(axis=0)
            total = table.sum()
            expected_min = (
                np.outer(row_sums, col_sums) / total if total > 0 else None
            )
            use_fisher = (
                total == 0
                or expected_min is None
                or np.any(expected_min < 5)
            )

            if use_fisher:
                odds_ratio, p_value = fisher_exact(table)
                test_used = "fisher_exact"
                statistic = odds_ratio
            else:
                chi2, p_value, dof, _ = chi2_contingency(table)
                test_used = "chi2_contingency"
                statistic = chi2

            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "class": class_name,
                "n_samples_true_class": n_samples,
                "recall_a": round(a_correct / n_samples, 4),
                "recall_b": round(b_correct / n_samples, 4),
                "test_used": test_used,
                "statistic": round(float(statistic), 4),
                "p_value": p_value,
                "significant_at_0.05": bool(p_value < ALPHA),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 3. Bootstrap confidence intervals
# =============================================================================

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, alpha=0.05,
                  **metric_kwargs):
    """
    Generic bootstrap CI for a given metric function (e.g. accuracy_score,
    f1_score with average='macro').
    """
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx], **metric_kwargs)
        except Exception:
            continue
        scores.append(score)

    scores = np.array(scores)
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    point_estimate = metric_fn(y_true, y_pred, **metric_kwargs)
    return point_estimate, lower, upper


def run_bootstrap_summary(data, n_bootstrap=N_BOOTSTRAP):
    rows = []
    for model_name, df in data.items():
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        acc_point, acc_lo, acc_hi = bootstrap_ci(
            y_true, y_pred, accuracy_score, n_bootstrap=n_bootstrap
        )
        f1_point, f1_lo, f1_hi = bootstrap_ci(
            y_true, y_pred, f1_score, n_bootstrap=n_bootstrap,
            average="macro", zero_division=0
        )

        rows.append({
            "model": model_name,
            "accuracy_point": round(acc_point, 4),
            "accuracy_95ci_lower": round(acc_lo, 4),
            "accuracy_95ci_upper": round(acc_hi, 4),
            "macro_f1_point": round(f1_point, 4),
            "macro_f1_95ci_lower": round(f1_lo, 4),
            "macro_f1_95ci_upper": round(f1_hi, 4),
            "n_bootstrap_iterations": n_bootstrap,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading predictions from:", RESULTS_DIR)
    data = load_predictions(RESULTS_DIR, MODEL_FILES)
    print("Loaded models:", list(data.keys()))

    print("\nRunning McNemar pairwise tests...")
    mcnemar_df = run_mcnemar_pairwise(data)
    print(mcnemar_df.to_string(index=False))

    print("\nRunning per-class Chi-square / Fisher's exact tests...")
    per_class_df = run_per_class_tests(data)
    print(per_class_df.to_string(index=False))

    print("\nRunning bootstrap confidence intervals...")
    bootstrap_df = run_bootstrap_summary(data)
    print(bootstrap_df.to_string(index=False))

    out_path = os.path.join(RESULTS_DIR, "statistical_tests_summary.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        mcnemar_df.to_excel(writer, sheet_name="mcnemar_pairwise", index=False)
        per_class_df.to_excel(writer, sheet_name="per_class_chi2_fisher", index=False)
        bootstrap_df.to_excel(writer, sheet_name="bootstrap_ci", index=False)

    print(f"\nAll results saved to: {out_path}")


if __name__ == "__main__":
    main()
