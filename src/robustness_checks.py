"""
Robustness checks for the gene-panel classifier on GSE62232.

1. Replicate structure: how many independent patients sit behind each class.
2. Leakage in the original split: do non-tumor test arrays share a patient
   with training arrays?
3. Sample-level vs patient-grouped cross-validation.
4. Random-panel baseline: does the curated 29-gene panel beat random
   29-gene panels drawn from the rest of the transcriptome?
5. Single-gene AUCs for the top-ranked panel genes.

Usage:
    python src/robustness_checks.py --n-random 30
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

sys.path.insert(0, os.path.dirname(__file__))
from diagnostic_model import TARGET_GENES, load_data  # noqa: E402

FORCED = "GSM1523338_JZN_0934N_U133_2.3.CEL.GZ"


def model():
    return CatBoostClassifier(iterations=150, verbose=0, random_seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expression", default="data/expression_matrix.csv")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--group-pattern", default=r"JZN_(\d+[NT])")
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--out", default="results/robustness_summary.txt")
    args = ap.parse_args()

    expr, labels = load_data(args.expression, args.labels)
    y = labels.iloc[:, 0].values
    pat = re.compile(args.group_pattern, re.IGNORECASE)
    groups = np.array([pat.search(s).group(1).upper() for s in expr.index])
    panel = [g for g in TARGET_GENES if g in expr.columns]
    X = expr[panel]
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    # 1. replicate structure
    log("== 1. Replicate structure ==")
    for cls in (0, 1):
        g = groups[y == cls]
        log(f"class {cls}: {len(g)} arrays from {len(set(g))} independent groups")
    log(f"class-0 arrays per group: {pd.Series(groups[y == 0]).value_counts().to_dict()}")

    # 2. leakage in the original (README v1) split
    log("\n== 2. Leakage in original split (seed 42, 30% test, forced sample) ==")
    mask = X.index == FORCED
    Xr, yr = X[~mask], y[~mask]
    tr_idx, te_idx = train_test_split(
        np.arange(len(yr)), test_size=0.3, random_state=42, stratify=yr
    )
    g_rest = groups[~mask]
    train_groups = set(g_rest[tr_idx]) | set(groups[mask])
    test_neg = [str(g) for g, lab in zip(g_rest[te_idx], yr[te_idx]) if lab == 0]
    leaked = [g for g in test_neg if g in train_groups]
    log(f"class-0 test arrays: {test_neg}")
    log(f"of which share a patient with training: {len(leaked)}/{len(test_neg)}")

    # 3. sample-level vs grouped CV
    log("\n== 3. CV AUC, curated panel (CatBoost, default params) ==")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    a = cross_val_score(model(), X, y, cv=skf, scoring="roc_auc")
    b = cross_val_score(model(), X, y, cv=sgkf, groups=groups, scoring="roc_auc")
    log(f"sample-level : {a.mean():.3f} ± {a.std():.3f}")
    log(f"patient-group: {b.mean():.3f} ± {b.std():.3f}")

    # 4. random-panel baseline
    log(f"\n== 4. Random 29-gene panels (n={args.n_random}), patient-grouped CV ==")
    rng = np.random.default_rng(0)
    pool = [g for g in expr.columns[expr.var() > 0] if g not in TARGET_GENES]
    rand = np.array([
        cross_val_score(model(), expr[list(rng.choice(pool, len(panel), replace=False))],
                        y, cv=sgkf, groups=groups, scoring="roc_auc").mean()
        for _ in range(args.n_random)
    ])
    log(f"random panels: mean {rand.mean():.3f}, median {np.median(rand):.3f}, "
        f"min {rand.min():.3f}, max {rand.max():.3f}")
    log(f"fraction of random panels with AUC >= curated ({b.mean():.3f}): "
        f"{(rand >= b.mean()).mean():.2f}")

    # 5. single genes
    log("\n== 5. Single-gene grouped-CV AUC ==")
    for g in ["SLC16A4", "BAX", "ENO1", "PKM", "SNAI1"]:
        if g in expr.columns:
            s = cross_val_score(model(), expr[[g]], y, cv=sgkf, groups=groups,
                                scoring="roc_auc").mean()
            log(f"{g:8s} {s:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
