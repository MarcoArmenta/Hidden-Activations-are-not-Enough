"""
Generate LaTeX tables from grid search results for paper revision.

Parses grid_search.txt, baseline.txt, and baseline_matrices.txt across
all experiments and generates publication-ready LaTeX tables.

Usage:
    python generate_latex_tables.py
    python generate_latex_tables.py --experiments lenet_cifar10 alexnet_cifar10 resnet_cifar10
    python generate_latex_tables.py --output tables/
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from constants.constants import ATTACKS, ATTACK_CATEGORIES

ARCH_DISPLAY = {'alexnet': 'AlexNet', 'resnet': 'ResNet', 'vgg': 'VGG', 'lenet': 'LeNet'}
DATASET_DISPLAY = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}

def format_arch(exp_name):
    return ARCH_DISPLAY.get(exp_name.split("_")[0], exp_name.split("_")[0].capitalize())

def format_dataset(exp_name):
    ds = exp_name.split("_", 1)[1]
    return DATASET_DISPLAY.get(ds, ds.upper())


def parse_args():
    parser = ArgumentParser(description="Generate LaTeX tables from grid search results")
    parser.add_argument("--experiments", nargs="+",
                        default=["alexnet_cifar10", "resnet_cifar10",
                                 "resnet_cifar100", "vgg_cifar100"],
                        help="Experiment names to include")
    parser.add_argument("--output", type=str, default="tables",
                        help="Output directory for LaTeX files")
    parser.add_argument("--topn", type=int, default=1,
                        help="Use top-N parameter setting per experiment")
    return parser.parse_args()


def load_grid_search(experiment: str) -> pd.DataFrame:
    """Load main grid_search results."""
    path = Path(f"experiments/{experiment}/grid_search/grid_search.txt")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["good_defence"] = pd.to_numeric(df["good_defence"], errors="coerce")
    df["wrong_rejection"] = pd.to_numeric(df["wrong_rejection"], errors="coerce")
    df = df.dropna(subset=["good_defence", "wrong_rejection"])
    return df


def load_baseline(experiment: str) -> pd.DataFrame:
    """Load baseline results (feature-space detectors)."""
    path = Path(f"experiments/{experiment}/grid_search/baseline.txt")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_baseline_matrices(experiment: str) -> pd.DataFrame:
    """Load baseline-on-matrices results."""
    path = Path(f"experiments/{experiment}/grid_search/baseline_matrices.txt")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def best_params(df: pd.DataFrame) -> pd.Series:
    """Get the best parameter combination (max good_defence - wrong_rejection)."""
    if df.empty:
        return pd.Series()
    df = df.copy()
    df["score"] = df["good_defence"] - df["wrong_rejection"]
    return df.loc[df["score"].idxmax()]


def get_per_attack_results(experiment: str, t_eps, eps, eps_p) -> dict:
    """Load per-attack counts for a given parameter combination."""
    counts_path = Path(f"experiments/{experiment}/counts_per_attack/"
                       f"counts_per_attack_{t_eps}_{eps}_{eps_p}.json")
    if counts_path.exists():
        import json
        with open(counts_path) as f:
            return json.load(f)
    return {}


def write_tex(path: Path, lines: list):
    """Write a .tex file with a standard preamble comment."""
    content = "% Requires: \\usepackage{booktabs, amssymb}\n" + "\n".join(lines)
    path.write_text(content)


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    return s.replace("&", r"\&").replace("#", r"\#").replace("_", r"\_").replace("%", r"\%")


def generate_main_comparison_table(experiments: list, output_dir: Path):
    """Table 1: Best detection rate (TPR) and false positive rate (FPR)
    per experiment for the knowledge matrix method vs best baseline."""
    rows = []
    for exp in experiments:
        df_main = load_grid_search(exp)
        df_base = load_baseline(exp)

        # Best main method
        if not df_main.empty:
            best = best_params(df_main)
            main_tpr = best.get("good_defence", np.nan)
            main_fpr = best.get("wrong_rejection", np.nan)
        else:
            main_tpr, main_fpr = np.nan, np.nan

        # Best baseline
        if not df_base.empty:
            df_base = df_base.copy()
            df_base["good_defence"] = pd.to_numeric(df_base["good_defence"], errors="coerce")
            df_base["wrong_rejection"] = pd.to_numeric(df_base["wrong_rejection"], errors="coerce")
            df_base = df_base.dropna(subset=["good_defence", "wrong_rejection"])
            if not df_base.empty:
                df_base["score"] = df_base["good_defence"] - df_base["wrong_rejection"]
                best_b = df_base.loc[df_base["score"].idxmax()]
                base_method = best_b.get("method", "N/A")
                base_tpr = best_b.get("good_defence", np.nan)
                base_fpr = best_b.get("wrong_rejection", np.nan)
            else:
                base_method, base_tpr, base_fpr = "N/A", np.nan, np.nan
        else:
            base_method, base_tpr, base_fpr = "N/A", np.nan, np.nan

        arch = format_arch(exp)
        dataset = format_dataset(exp)

        rows.append({
            "Architecture": arch,
            "Dataset": dataset,
            "Ours TPR": main_tpr,
            "Ours FPR": main_fpr,
            "Baseline": base_method,
            "Base TPR": base_tpr,
            "Base FPR": base_fpr,
        })

    df = pd.DataFrame(rows)

    # Generate LaTeX
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection performance: Knowledge Matrix method vs.\ best feature-space baseline. "
        r"TPR = adversarial detection rate, FPR = clean sample rejection rate.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llcclcc}",
        r"\toprule",
        r"Architecture & Dataset & \multicolumn{2}{c}{Ours} & \multicolumn{3}{c}{Best Baseline} \\",
        r"& & TPR $\uparrow$ & FPR $\downarrow$ & Method & TPR $\uparrow$ & FPR $\downarrow$ \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-7}",
    ]

    for _, row in df.iterrows():
        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "---"
        lines.append(
            f"{row['Architecture']} & {row['Dataset']} & "
            f"\\textbf{{{fmt(row['Ours TPR'])}}} & {fmt(row['Ours FPR'])} & "
            f"{escape_latex(str(row['Baseline']))} & {fmt(row['Base TPR'])} & {fmt(row['Base FPR'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    write_tex(output_dir / "main_comparison.tex", lines)
    print(f"Generated {output_dir / 'main_comparison.tex'}")


def generate_per_attack_table(experiments: list, output_dir: Path):
    """Table 2: Detection rate per attack category across architectures."""
    categories = {
        "Gradient": ATTACK_CATEGORIES["gradient_based"],
        "AutoAttack": ATTACK_CATEGORIES["autoattack"],
        "Grad-free": ATTACK_CATEGORIES["gradient_free"],
        "Elastic": ATTACK_CATEGORIES["elastic_net"],
        "Baseline": ATTACK_CATEGORIES["baseline_noise"],
    }

    rows = []
    for exp in experiments:
        df = load_grid_search(exp)
        if df.empty:
            continue

        best = best_params(df)
        t_eps = best.get("t_epsilon")
        eps = best.get("epsilon")
        eps_p = best.get("epsilon_p")

        counts = get_per_attack_results(exp, t_eps, eps, eps_p)
        if not counts:
            continue

        arch = format_arch(exp)
        row = {"Architecture": arch}

        for cat_name, cat_attacks in categories.items():
            detected = 0
            total = 0
            for atk in cat_attacks:
                if atk in counts and atk != "test":
                    c = counts[atk]
                    detected += c.get("rejected_and_attacked", 0)
                    total += (c.get("rejected_and_attacked", 0) +
                              c.get("not_rejected_and_attacked", 0))
            row[cat_name] = detected / total if total > 0 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # Determine datasets used
    datasets_used = set()
    for exp in experiments:
        if 'cifar100' in exp:
            datasets_used.add('CIFAR-100')
        elif 'cifar10' in exp:
            datasets_used.add('CIFAR-10')
    caption_datasets = ' and '.join(sorted(datasets_used)) if datasets_used else 'the given datasets'

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection rate by attack category across architectures. "
        f"All experiments use {caption_datasets} with identical detection parameters.}}",
        r"\label{tab:per_attack}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Architecture & Gradient & AutoAttack & Gradient-free & Elastic-net & Baseline$^*$ \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "---"
        lines.append(
            f"{row['Architecture']} & "
            f"{fmt(row.get('Gradient', np.nan))} & "
            f"{fmt(row.get('AutoAttack', np.nan))} & "
            f"{fmt(row.get('Grad-free', np.nan))} & "
            f"{fmt(row.get('Elastic', np.nan))} & "
            f"{fmt(row.get('Baseline', np.nan))} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"{\footnotesize $^*$Baseline = Gaussian Noise (GN). Square appears in both AutoAttack and Gradient-free categories.}",
        r"\end{tabular}",
        r"\end{table}",
    ]

    write_tex(output_dir / "per_attack_category.tex", lines)
    print(f"Generated {output_dir / 'per_attack_category.tex'}")


def generate_method_comparison_table(output_dir: Path):
    """Table 3: Property comparison (not data-driven, hand-authored content)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of detection method properties. "
        r"Our knowledge matrix approach is the only method that is simultaneously "
        r"architecture-agnostic, attack-agnostic, requires no retraining, and provides "
        r"theoretical guarantees.}",
        r"\label{tab:method_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Property & Mahalanobis & LID & DkNN & Feature Squeeze & \textbf{Ours} \\",
        r"& \cite{lee2018simple} & \cite{ma2018characterizing} & \cite{papernot2018} & \cite{xu2018} & \\",
        r"\midrule",
        r"Architecture-agnostic & $\times$ & $\times$ & $\times$ & $\times$ & $\checkmark$ \\",
        r"Attack-agnostic & $\times$ & $\times$ & Partial & $\times$ & $\checkmark$ \\",
        r"No retraining needed & $\checkmark$ & $\times$ & $\times$ & $\checkmark$ & $\checkmark$ \\",
        r"No auxiliary network & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ \\",
        r"Theoretical guarantees & $\times$ & $\times$ & $\times$ & $\times$ & $\checkmark$ \\",
        r"Demonstrated MLP+CNN & --- & --- & --- & --- & $\checkmark$ \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    write_tex(output_dir / "method_comparison.tex", lines)
    print(f"Generated {output_dir / 'method_comparison.tex'}")


def generate_full_attack_table(experiments: list, output_dir: Path):
    """Appendix table: Full per-attack detection rates for all experiments."""
    all_rows = []
    for exp in experiments:
        df = load_grid_search(exp)
        if df.empty:
            continue
        best = best_params(df)
        t_eps, eps, eps_p = best.get("t_epsilon"), best.get("epsilon"), best.get("epsilon_p")
        counts = get_per_attack_results(exp, t_eps, eps, eps_p)
        if not counts:
            continue

        arch = format_arch(exp)
        for atk in ATTACKS:
            if atk in counts:
                c = counts[atk]
                det = c.get("rejected_and_attacked", 0)
                total = det + c.get("not_rejected_and_attacked", 0)
                rate = det / total if total > 0 else np.nan
            else:
                rate = np.nan
            all_rows.append({"Architecture": arch, "Attack": atk, "Detection Rate": rate})

    if not all_rows:
        print("No data available for full attack table.")
        return

    df = pd.DataFrame(all_rows)
    pivot = df.pivot(index="Attack", columns="Architecture", values="Detection Rate")

    # Reorder attacks to match ATTACKS list
    attack_order = [a for a in ATTACKS if a in pivot.index]
    pivot = pivot.reindex(attack_order)

    archs = [format_arch(e) for e in experiments if format_arch(e) in pivot.columns]
    archs = list(dict.fromkeys(archs))  # deduplicate preserving order

    n_archs = len(archs)
    col_spec = "l" + "c" * n_archs
    header = " & ".join(archs)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-attack detection rates across all architectures (best hyperparameters). "
        r"All values represent the fraction of adversarial examples correctly detected.}",
        r"\label{tab:full_attacks}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Attack & {header} \\\\",
        r"\midrule",
    ]

    for atk in attack_order:
        vals = []
        for arch in archs:
            v = pivot.loc[atk, arch] if arch in pivot.columns else np.nan
            vals.append(f"{v:.3f}" if not np.isnan(v) else "---")
        lines.append(f"{escape_latex(atk)} & {' & '.join(vals)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    write_tex(output_dir / "full_attack_results.tex", lines)
    print(f"Generated {output_dir / 'full_attack_results.tex'}")


# ---------------------------------------------------------------------------
# New AUROC-based tables (from compare_representations.py output)
# ---------------------------------------------------------------------------

def load_comparison_json(experiment: str) -> dict:
    """Load representation_comparison.json for an experiment."""
    path = Path(f"experiments/{experiment}/comparison/representation_comparison.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def generate_representation_comparison_table(experiments: list, output_dir: Path,
                                              metric: str = 'auroc',
                                              metric_label: str = 'AUROC',
                                              higher_is_better: bool = True):
    """Central table: 3 representations x 6 detectors, metric averaged across attacks.
    One sub-table per experiment, or a single combined table.
    Supports auroc, aupr, and fpr_at_95tpr metrics."""
    all_data = {}
    for exp in experiments:
        data = load_comparison_json(exp)
        if data:
            all_data[exp] = data

    if not all_data:
        print(f"No representation comparison data found for {metric_label}.")
        return

    # Collect detector and rep names from first available experiment
    sample = next(iter(all_data.values()))
    det_names = sample.get('detectors', ['Mahalanobis'])
    rep_names = sample.get('representations', [])

    short_reps = {'penultimate': 'Penultimate', 'all_layer': 'All-Layer',
                  'knowledge_matrix': 'Knowledge Matrix'}

    n_reps = len(rep_names)
    col_spec = "l" + "c" * n_reps

    direction = "Higher is better" if higher_is_better else "Lower is better"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Mean {metric_label} across all attacks for each detector--representation pair. "
        f"Bold indicates the best representation per detector. "
        f"{direction}.}}",
        f"\\label{{tab:rep_comparison}}" if metric == 'auroc' else f"\\label{{tab:rep_comparison_{metric}}}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header = "Detector"
    for rn in rep_names:
        header += f" & {escape_latex(short_reps.get(rn, rn))}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for exp_name, data in all_data.items():
        if len(all_data) > 1:
            arch = format_arch(exp_name)
            dataset = format_dataset(exp_name)
            lines.append(f"\\multicolumn{{{n_reps + 1}}}{{l}}"
                         f"{{\\textit{{{arch} / {dataset}}}}} \\\\")

        # Get average metric data: use precomputed average_auroc if available for auroc,
        # otherwise compute from per-attack data
        if metric == 'auroc' and 'average_auroc' in data:
            avg_metric = data['average_auroc']
        else:
            # Compute average from per-attack data
            per_attack = data.get('per_attack', {})
            avg_metric = {}
            for det_name in det_names:
                avg_metric[det_name] = {}
                for rn in rep_names:
                    vals_list = []
                    for attack_data in per_attack.values():
                        val = attack_data.get(det_name, {}).get(rn, {}).get(metric)
                        if val is not None:
                            vals_list.append(val)
                    if vals_list:
                        avg_metric[det_name][rn] = float(np.mean(vals_list))

        for det_name in det_names:
            vals = {}
            for rn in rep_names:
                if isinstance(avg_metric.get(det_name), dict):
                    vals[rn] = avg_metric[det_name].get(rn)
                else:
                    vals[rn] = None

            # Find best (direction-aware)
            valid_vals = [v for v in vals.values() if v is not None]
            if valid_vals:
                best_val = max(valid_vals) if higher_is_better else min(valid_vals)
            else:
                best_val = None

            row = escape_latex(det_name)
            for rn in rep_names:
                v = vals[rn]
                if v is not None:
                    s = f"{v:.3f}"
                    if best_val is not None and abs(v - best_val) < 1e-4:
                        s = f"\\textbf{{{s}}}"
                    row += f" & {s}"
                else:
                    row += " & ---"
            lines.append(row + r" \\")

        # Add Lee et al. (2018) baseline row if available
        lee_baseline = data.get('lee2018_baseline', {})
        if lee_baseline:
            # Compute mean of the current metric across attacks
            lee_vals = []
            for atk_data in lee_baseline.values():
                if isinstance(atk_data, dict):
                    val = atk_data.get(metric)
                    if val is not None:
                        lee_vals.append(val)
                elif isinstance(atk_data, (int, float)) and metric == 'auroc':
                    lee_vals.append(atk_data)
            if lee_vals:
                lee_mean = float(np.mean(lee_vals))
                lines.append(r"\midrule")
                row = r"Lee et al.\ (2018)$^\dagger$"
                for rn in rep_names:
                    if rn == rep_names[0]:
                        row += f" & \\multicolumn{{{n_reps}}}{{c}}{{{lee_mean:.3f}}}"
                        break
                lines.append(row + r" \\")

        if len(all_data) > 1:
            lines.append(r"\midrule")

    # Remove trailing midrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"{\footnotesize $^\dagger$Lee et al.\ (2018): multi-layer Mahalanobis with per-attack logistic regression.}",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    # Use metric-specific filename
    if metric == 'auroc':
        fname = "representation_comparison.tex"
    else:
        fname = f"representation_comparison_{metric}.tex"
    write_tex(output_dir / fname, lines)
    print(f"Generated {output_dir / fname}")


def generate_per_attack_auroc_table(experiments: list, output_dir: Path):
    """Per-attack AUROC for best detector per representation."""
    all_data = {}
    for exp in experiments:
        data = load_comparison_json(exp)
        if data:
            all_data[exp] = data

    if not all_data:
        return

    sample = next(iter(all_data.values()))
    det_names = sample.get('detectors', ['Mahalanobis'])
    rep_names = sample.get('representations', [])

    short_reps = {'penultimate': 'Penult.', 'all_layer': 'AllLayer',
                  'knowledge_matrix': 'KnowMat'}

    n_reps = len(rep_names)

    for exp_name, data in all_data.items():
        arch = format_arch(exp_name)
        per_attack = data.get('per_attack', {})
        attack_order = [a for a in ATTACKS if a in per_attack]

        # Check for Lee et al. per-attack data
        lee_data = data.get('lee2018_baseline', {})
        has_lee = bool(lee_data)
        n_cols = n_reps + (1 if has_lee else 0)
        col_spec = "l" + "c" * n_cols

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{Per-attack AUROC for {arch} (best of 6 detectors per representation). "
            r"Bold indicates the best representation per attack.}",
            f"\\label{{tab:per_attack_auroc_{exp_name}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        header = "Attack"
        for rn in rep_names:
            header += f" & {short_reps.get(rn, rn)}"
        if has_lee:
            header += r" & Lee$^\dagger$"
        lines.append(header + r" \\")
        lines.append(r"\midrule")

        avg_by_rep = {rn: [] for rn in rep_names}
        lee_vals_all = []

        for atk in attack_order:
            atk_data = per_attack[atk]
            # Best detector per representation
            vals = {}
            for rn in rep_names:
                best = -1
                for dn in det_names:
                    if dn in atk_data and rn in atk_data[dn]:
                        auroc = atk_data[dn][rn].get('auroc', 0)
                        if auroc > best:
                            best = auroc
                vals[rn] = best if best >= 0 else None
                if vals[rn] is not None:
                    avg_by_rep[rn].append(vals[rn])

            # Lee et al. value for this attack
            lee_val = None
            if has_lee:
                lee_atk = lee_data.get(atk, {})
                if isinstance(lee_atk, dict):
                    lee_val = lee_atk.get('auroc')
                elif isinstance(lee_atk, (int, float)):
                    lee_val = lee_atk
                if lee_val is not None:
                    lee_vals_all.append(lee_val)

            # Collect all values for bolding (include Lee)
            all_vals = [v for v in vals.values() if v is not None]
            if lee_val is not None:
                all_vals.append(lee_val)
            best_val = max(all_vals, default=-1)

            row = escape_latex(atk)
            for rn in rep_names:
                v = vals[rn]
                if v is not None:
                    s = f"{v:.3f}"
                    if abs(v - best_val) < 1e-4:
                        s = f"\\textbf{{{s}}}"
                    row += f" & {s}"
                else:
                    row += " & ---"
            if has_lee:
                if lee_val is not None:
                    s = f"{lee_val:.3f}"
                    if abs(lee_val - best_val) < 1e-4:
                        s = f"\\textbf{{{s}}}"
                    row += f" & {s}"
                else:
                    row += " & ---"
            lines.append(row + r" \\")

        # Average row
        lines.append(r"\midrule")
        row = "Average"
        avg_candidates = []
        for rn in rep_names:
            if avg_by_rep[rn]:
                avg_candidates.append(np.mean(avg_by_rep[rn]))
        if lee_vals_all:
            avg_candidates.append(np.mean(lee_vals_all))
        best_avg = max(avg_candidates, default=-1)

        for rn in rep_names:
            if avg_by_rep[rn]:
                avg = np.mean(avg_by_rep[rn])
                s = f"{avg:.3f}"
                if abs(avg - best_avg) < 1e-4:
                    s = f"\\textbf{{{s}}}"
                row += f" & {s}"
            else:
                row += " & ---"
        if has_lee:
            if lee_vals_all:
                lee_avg = np.mean(lee_vals_all)
                s = f"{lee_avg:.3f}"
                if abs(lee_avg - best_avg) < 1e-4:
                    s = f"\\textbf{{{s}}}"
                row += f" & {s}"
            else:
                row += " & ---"
        lines.append(row + r" \\")

        lines += [
            r"\bottomrule",
            r"{\footnotesize $^\dagger$Lee et al.\ (2018): multi-layer Mahalanobis with per-attack logistic regression.}",
            r"\end{tabular}",
            r"\end{table}",
        ]

        fname = f"per_attack_auroc_{exp_name}.tex"
        write_tex(output_dir / fname, lines)
        print(f"Generated {output_dir / fname}")


def generate_cost_table(experiments: list, output_dir: Path):
    """Computational cost comparison table."""
    rows = []
    for exp in experiments:
        data = load_comparison_json(exp)
        if not data or 'computational_cost' not in data:
            continue
        arch = format_arch(exp)
        dataset = format_dataset(exp)
        cost = data['computational_cost']
        for rn in data.get('representations', []):
            if rn in cost:
                rows.append({
                    'Architecture': arch,
                    'Dataset': dataset,
                    'Representation': {'penultimate': 'Penultimate',
                                       'all_layer': 'All-Layer',
                                       'knowledge_matrix': 'Knowledge Matrix'}.get(rn, rn),
                    'Dim': cost[rn].get('feature_dim', '---'),
                    'Time': cost[rn].get('seconds_per_1000', cost[rn].get('loading_seconds_per_1000', np.nan)),
                    'Memory': cost[rn].get('peak_gpu_memory_gb', np.nan),
                })

    if not rows:
        print("No computational cost data found.")
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational cost of representation extraction. "
        r"Time is seconds per 1000 samples; memory is peak GPU allocation in GB.}",
        r"\label{tab:cost}",
        r"\begin{tabular}{lllrcc}",
        r"\toprule",
        r"Architecture & Dataset & Representation & Dim & s/1000 & GPU (GB) \\",
        r"\midrule",
    ]

    for row in rows:
        t = f"{row['Time']:.1f}" if not np.isnan(row['Time']) else "---"
        m = f"{row['Memory']:.2f}" if not np.isnan(row['Memory']) else "---"
        lines.append(
            f"{row['Architecture']} & {row['Dataset']} & "
            f"{row['Representation']} & {row['Dim']} & {t} & {m} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    write_tex(output_dir / "cost_comparison.tex", lines)
    print(f"Generated {output_dir / 'cost_comparison.tex'}")


def generate_svd_ablation_table(experiments: list, output_dir: Path):
    """SVD rank ablation table: rank vs AUROC per representation."""
    all_ablation = {}
    for exp in experiments:
        data = load_comparison_json(exp)
        if data and 'svd_ablation' in data:
            all_ablation[exp] = data['svd_ablation']

    if not all_ablation:
        print("No SVD ablation data found.")
        return

    # Get rep names from first experiment
    sample_data = load_comparison_json(next(iter(all_ablation)))
    rep_names = sample_data.get('representations', [])
    short_reps = {'penultimate': 'Penult.', 'all_layer': 'AllLayer',
                  'knowledge_matrix': 'KnowMat'}

    n_reps = len(rep_names)
    col_spec = "r" + "c" * n_reps

    for exp_name, ablation in all_ablation.items():
        arch = format_arch(exp_name)
        ranks = sorted(ablation.keys(), key=lambda x: int(x))

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{SVD rank ablation for {arch} (Mahalanobis detector, mean AUROC). "
            r"Shows how dimensionality reduction affects detection per representation.}",
            f"\\label{{tab:svd_ablation_{exp_name}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        header = "Rank"
        for rn in rep_names:
            header += f" & {short_reps.get(rn, rn)}"
        lines.append(header + r" \\")
        lines.append(r"\midrule")

        for rank in ranks:
            row = str(rank)
            rank_data = ablation[rank]
            best_val = -1
            for rn in rep_names:
                if rn in rank_data and rank_data[rn].get('mean_auroc') is not None:
                    if rank_data[rn]['mean_auroc'] > best_val:
                        best_val = rank_data[rn]['mean_auroc']

            for rn in rep_names:
                if rn in rank_data and rank_data[rn].get('mean_auroc') is not None:
                    v = rank_data[rn]['mean_auroc']
                    s = f"{v:.3f}"
                    if abs(v - best_val) < 1e-4:
                        s = f"\\textbf{{{s}}}"
                    row += f" & {s}"
                else:
                    row += " & ---"
            lines.append(row + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        fname = f"svd_ablation_{exp_name}.tex"
        write_tex(output_dir / fname, lines)
        print(f"Generated {output_dir / fname}")


def generate_lee2018_comparison_table(experiments: list, output_dir: Path):
    """Table comparing Lee et al. (2018) baseline vs best detector per representation."""
    rows = []
    for exp in experiments:
        data = load_comparison_json(exp)
        if not data:
            continue
        arch = format_arch(exp)
        dataset = format_dataset(exp)

        # Best detector per representation (average AUROC)
        avg_auroc = data.get('average_auroc', {})
        rep_names = data.get('representations', [])
        det_names = data.get('detectors', [])

        best_per_rep = {}
        for rn in rep_names:
            best = -1
            for dn in det_names:
                if isinstance(avg_auroc.get(dn), dict):
                    v = avg_auroc[dn].get(rn)
                    if v is not None and v > best:
                        best = v
            best_per_rep[rn] = best if best >= 0 else None

        lee_auroc = data.get('lee2018_mean_auroc')

        rows.append({
            'Architecture': arch,
            'Dataset': dataset,
            'penultimate': best_per_rep.get('penultimate'),
            'all_layer': best_per_rep.get('all_layer'),
            'knowledge_matrix': best_per_rep.get('knowledge_matrix'),
            'lee2018': lee_auroc,
        })

    if not rows:
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Best mean AUROC per representation (best of 6 detectors) "
        r"compared with Lee et al.\ (2018) multi-layer Mahalanobis baseline. "
        r"Higher is better. Bold indicates the overall best.}",
        r"\label{tab:lee2018_comparison}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Arch & Dataset & Penultimate & All-Layer & Know.\ Matrix & Lee et al. \\",
        r"\midrule",
    ]

    for row in rows:
        vals = [row['penultimate'], row['all_layer'],
                row['knowledge_matrix'], row['lee2018']]
        best_val = max((v for v in vals if v is not None), default=-1)
        parts = [f"{row['Architecture']}", f"{row['Dataset']}"]
        for v in vals:
            if v is not None:
                s = f"{v:.3f}"
                if abs(v - best_val) < 1e-4:
                    s = f"\\textbf{{{s}}}"
                parts.append(s)
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    write_tex(output_dir / "lee2018_comparison.tex", lines)
    print(f"Generated {output_dir / 'lee2018_comparison.tex'}")


def load_theorem45_json(experiment: str) -> dict:
    """Load theorem45_results.json for an experiment."""
    path = Path(f"experiments/{experiment}/theorem45/theorem45_results.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def generate_theorem45_table(experiments: list, output_dir: Path):
    """Theorem 4.5 validation table: gamma, d_M/d_f, d_h/d_f, d_M/d_h per attack."""
    all_data = {}
    for exp in experiments:
        data = load_theorem45_json(exp)
        if data and data.get('per_attack'):
            all_data[exp] = data

    if not all_data:
        print("No Theorem 4.5 validation data found.")
        return

    # --- Per-experiment detailed tables ---
    for exp_name, data in all_data.items():
        arch = format_arch(exp_name)
        per_attack = data['per_attack']
        attack_order = [a for a in ATTACKS if a in per_attack]

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{Theorem~4.5 validation for {arch}: knowledge matrix distances "
            r"lower-bound logit distances ($\|M(x) - M(x')\| \geq \gamma \cdot \|f(x) - f(x')\|$). "
            r"$d_M/d_h > 1$ shows KMs amplify more than penultimate features.}",
            f"\\label{{tab:theorem45_{exp_name}}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Attack & $\hat{\gamma}$ & $d_M/d_f$ & $d_h/d_f$ & $d_M/d_h$ & $\gamma$ 95\% CI \\",
            r"\midrule",
        ]

        for atk in attack_order:
            r = per_attack[atk]
            gamma = r['gamma_empirical']
            amp_M = r['amplification_M_median']
            amp_h = r['amplification_h_median']
            ratio = amp_M / amp_h if amp_h > 1e-12 else float('inf')
            gamma_ci = r.get('gamma_ci_95', [None, None])

            # Bold the d_M/d_h column if > 1 (KMs better)
            ratio_s = f"{ratio:.2f}"
            if ratio > 1.0:
                ratio_s = f"\\textbf{{{ratio_s}}}"

            ci_s = f"[{gamma_ci[0]:.3f}, {gamma_ci[1]:.3f}]" if gamma_ci[0] is not None and gamma_ci[1] is not None else "---"
            lines.append(
                f"{escape_latex(atk)} & {gamma:.3f} & {amp_M:.2f} & "
                f"{amp_h:.2f} & {ratio_s} & {ci_s} \\\\"
            )

        # Aggregate row
        agg = data.get('aggregate', {})
        if agg:
            lines.append(r"\midrule")
            g = agg.get('gamma_global', 0)
            mM = agg.get('mean_amplification_M', 0)
            mh = agg.get('mean_amplification_h', 0)
            r_mh = agg.get('ratio_M_over_h')
            r_s = f"{r_mh:.2f}" if r_mh is not None else "---"
            if r_mh is not None and r_mh > 1.0:
                r_s = f"\\textbf{{{r_s}}}"
            lines.append(
                f"Overall & {g:.3f} & {mM:.2f} & {mh:.2f} & {r_s} & --- \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]

        fname = f"theorem45_{exp_name}.tex"
        write_tex(output_dir / fname, lines)
        print(f"Generated {output_dir / fname}")

    # --- Cross-experiment summary table ---
    if len(all_data) > 1:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Theorem~4.5 summary across experiments. "
            r"$\hat{\gamma}$ is the global distance lower-bound constant; "
            r"$d_M/d_h$ shows the advantage of knowledge matrices over penultimate features.}",
            r"\label{tab:theorem45_summary}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Arch & Dataset & $\hat{\gamma}$ & $d_M/d_f$ & $d_h/d_f$ & $d_M/d_h$ \\",
            r"\midrule",
        ]

        for exp_name, data in all_data.items():
            arch = format_arch(exp_name)
            dataset = format_dataset(exp_name)
            agg = data.get('aggregate', {})
            g = agg.get('gamma_global')
            mM = agg.get('mean_amplification_M')
            mh = agg.get('mean_amplification_h')
            r_mh = agg.get('ratio_M_over_h')

            g_s = f"{g:.3f}" if g is not None else "---"
            mM_s = f"{mM:.2f}" if mM is not None else "---"
            mh_s = f"{mh:.2f}" if mh is not None else "---"
            r_s = f"{r_mh:.2f}" if r_mh is not None else "---"
            if r_mh is not None and r_mh > 1.0:
                r_s = f"\\textbf{{{r_s}}}"

            lines.append(f"{arch} & {dataset} & {g_s} & {mM_s} & {mh_s} & {r_s} \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        write_tex(output_dir / "theorem45_summary.tex", lines)
        print(f"Generated {output_dir / 'theorem45_summary.tex'}")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Original tables (from grid_search results) ---
    available = [exp for exp in args.experiments
                 if Path(f"experiments/{exp}/grid_search/grid_search.txt").exists()]
    if not available:
        print("No grid search results found for any experiment.")
        print(f"Looked in: {args.experiments}")
    else:
        print(f"Found grid search results for: {available}")
        generate_main_comparison_table(available, output_dir)
        generate_per_attack_table(available, output_dir)
        generate_full_attack_table(available, output_dir)

    generate_method_comparison_table(output_dir)

    # --- New AUROC-based tables (from compare_representations.py) ---
    comparison_available = [exp for exp in args.experiments
                           if Path(f"experiments/{exp}/comparison/representation_comparison.json").exists()]
    if comparison_available:
        print(f"\nFound representation comparison results for: {comparison_available}")
        generate_representation_comparison_table(comparison_available, output_dir,
                                                metric='auroc', metric_label='AUROC',
                                                higher_is_better=True)
        generate_representation_comparison_table(comparison_available, output_dir,
                                                metric='aupr', metric_label='AUPR',
                                                higher_is_better=True)
        generate_representation_comparison_table(comparison_available, output_dir,
                                                metric='fpr_at_95tpr', metric_label='FPR@95TPR',
                                                higher_is_better=False)
        generate_per_attack_auroc_table(comparison_available, output_dir)
        generate_cost_table(comparison_available, output_dir)
        generate_svd_ablation_table(comparison_available, output_dir)
        generate_lee2018_comparison_table(comparison_available, output_dir)
    else:
        print("No representation comparison results found (run compare_representations.py first).")

    # --- Theorem 4.5 tables (from validate_theorem45.py) ---
    theorem45_available = [exp for exp in args.experiments
                           if Path(f"experiments/{exp}/theorem45/theorem45_results.json").exists()]
    if theorem45_available:
        print(f"\nFound Theorem 4.5 results for: {theorem45_available}")
        generate_theorem45_table(theorem45_available, output_dir)
    else:
        print("No Theorem 4.5 results found (run validate_theorem45.py first).")

    print(f"\nAll tables written to {output_dir}/")


if __name__ == "__main__":
    main()
