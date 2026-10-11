"""
Gene-panel diagnostic classifier.

Trains CatBoost and XGBoost classifiers on a curated 29-gene panel
(EMT, apoptosis, glycolysis / OXPHOS, proliferation, lactate transport,
inflammation) from a samples x genes expression matrix, with Bayesian
hyperparameter search, held-out evaluation, cross-validation and SHAP.

Usage:
    python src/diagnostic_model.py \
        --expression data/expression_matrix.csv \
        --labels data/labels.csv \
        --outdir results
"""

import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from tqdm import tqdm

TARGET_GENES = [
    # EMT
    "CDH1", "CDH2", "SNAI1", "SNAI2", "VIM", "ACTA2",
    # Apoptosis / p53 axis
    "TP53", "BAX", "BCL2", "CASP3", "CDKN1A",
    # Mitochondrial / OXPHOS
    "COX11", "ATP6V1H", "NDUFAF5", "NDUFA10", "ESRRA",
    # Glycolysis
    "PKM", "TPI1", "LDHA", "ENO1",
    # Lactate transport
    "SLC16A1", "SLC16A4",
    # Proliferation / immortality
    "MKI67", "TERT", "TOP1", "MYC",
    # Signalling / inflammation
    "FGFR1", "CCL2", "CXCL8",
]


class DiagnosticModel:
    def __init__(self, expression_matrix, labels, model_type="catboost",
                 n_iter=60, seed=42, outdir="results", n_jobs=-1):
        self.available_genes = [g for g in TARGET_GENES if g in expression_matrix.columns]
        if len(self.available_genes) < 5:
            raise ValueError("Too few target genes found in expression matrix.")

        self.model_type = model_type
        self.n_iter = n_iter
        self.seed = seed
        self.n_jobs = n_jobs
        self.outdir = outdir
        self.fig_dir = os.path.join(outdir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

        self.X = expression_matrix[self.available_genes]
        self.y = labels.values.ravel()

        if model_type == "catboost":
            self.model = CatBoostClassifier(iterations=150, verbose=0, random_seed=seed)
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=150, random_state=seed, eval_metric="logloss", verbosity=0
            )
        else:
            raise ValueError("Invalid model_type. Choose 'catboost' or 'xgboost'")

    # ------------------------------------------------------------------ split
    def preprocess(self, test_size=0.3, sample_to_train=""):
        """Stratified train/test split. Optionally force one sample into training."""
        if sample_to_train and sample_to_train in self.X.index:
            mask = self.X.index == sample_to_train
            X_rest, y_rest = self.X.loc[~mask], self.y[~mask]
            X_tr, self.X_test, y_tr, self.y_test = train_test_split(
                X_rest, y_rest, test_size=test_size,
                random_state=self.seed, stratify=y_rest,
            )
            self.X_train = pd.concat([self.X.loc[mask], X_tr])
            self.y_train = pd.concat(
                [pd.Series(self.y[mask]), pd.Series(y_tr)]
            ).values
            print(f"Sample '{sample_to_train}' placed in the training set.")
        else:
            if sample_to_train:
                print(f"Warning: sample '{sample_to_train}' not found; standard split.")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size,
                random_state=self.seed, stratify=self.y,
            )

    # ----------------------------------------------------------------- tuning
    def _tune_hyperparameters(self):
        if self.model_type == "catboost":
            space = {
                "depth": Integer(4, 10),
                "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
                "l2_leaf_reg": Real(1, 10, prior="log-uniform"),
                "bagging_temperature": Real(0.5, 2.0),
            }
        else:
            space = {
                "max_depth": Integer(3, 9),
                "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
                "subsample": Real(0.6, 1.0),
                "colsample_bytree": Real(0.6, 1.0),
                "reg_lambda": Real(1, 10, prior="log-uniform"),
            }

        pbar = tqdm(total=self.n_iter, desc=f"Tuning {self.model_type}")
        search = BayesSearchCV(
            estimator=self.model,
            search_spaces=space,
            n_iter=self.n_iter,
            scoring="roc_auc",
            cv=5,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            optimizer_kwargs={"n_initial_points": min(20, self.n_iter)},
            verbose=0,
        )
        search.fit(self.X_train, self.y_train, callback=[lambda _: pbar.update(1)])
        pbar.close()
        self.model = search.best_estimator_
        self.best_params = dict(search.best_params_)
        print(f"Best {self.model_type.upper()} params: {self.best_params}")

    # ------------------------------------------------------------------ train
    def train(self, tune=True):
        self.best_params = {}
        if tune:
            self._tune_hyperparameters()
        self.model.fit(self.X_train, self.y_train)

        importances = (
            self.model.get_feature_importance()
            if self.model_type == "catboost"
            else self.model.feature_importances_
        )
        self.importance_df = pd.DataFrame(
            {"Gene": self.X_train.columns, "Importance": importances}
        ).sort_values("Importance", ascending=False)
        self.importance_df.to_csv(
            os.path.join(self.outdir, f"{self.model_type}_gene_importance.csv"), index=False
        )
        print(f"\nGene importance ({self.model_type.upper()}):")
        print(self.importance_df.head(10).to_string(index=False))

    # --------------------------------------------------------------- evaluate
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        self.test_auc = roc_auc_score(self.y_test, y_prob)

        print("\nClassification report:\n", classification_report(self.y_test, y_pred, zero_division=0))
        print("Test AUC:", round(self.test_auc, 3))
        print("Confusion matrix:\n", confusion_matrix(self.y_test, y_pred))

        self._plot_roc(y_prob)
        self._plot_pr(y_prob)
        self._plot_cm(y_pred)

    def cross_validate(self, cv=5):
        scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring="roc_auc")
        self.cv_scores = scores
        print(f"\nCV AUC (k={cv}): {scores.mean():.3f} ± {scores.std():.3f}  folds={scores.round(3)}")

    def explain_model(self, max_display=20):
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(self.X_train)
        plt.figure()
        shap.summary_plot(sv, self.X_train, plot_type="bar", max_display=max_display, show=False)
        self._save(f"{self.model_type}_shap.png")

    def classify_new_samples(self, new_samples):
        missing = set(self.available_genes) - set(new_samples.columns)
        if missing:
            raise ValueError(f"New samples missing {len(missing)} panel genes: {sorted(missing)}")
        X_new = new_samples[self.available_genes]
        return pd.DataFrame({
            "Sample_ID": new_samples.index,
            "Predicted_Class": self.model.predict(X_new),
            "Probability": self.model.predict_proba(X_new)[:, 1],
        })

    def run_full_pipeline(self, tune=True, sample_to_train=""):
        print(f"\n=== {self.model_type.upper()} pipeline ===")
        self.preprocess(sample_to_train=sample_to_train)
        self.train(tune=tune)
        self.evaluate()
        self.cross_validate()
        self.explain_model()

    # ------------------------------------------------------------------ plots
    def _save(self, name):
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, name), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_roc(self, y_prob):
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({self.model_type})"); plt.legend(); plt.grid(True)
        self._save(f"{self.model_type}_roc.png")

    def _plot_pr(self, y_prob):
        p, r, _ = precision_recall_curve(self.y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(r, p)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({self.model_type})"); plt.grid(True)
        self._save(f"{self.model_type}_pr.png")

    def _plot_cm(self, y_pred):
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt="d",
                    cmap="Blues", cbar=False)
        plt.xlabel("Predicted Label"); plt.ylabel("True Label")
        plt.title(f"Confusion Matrix ({self.model_type})")
        self._save(f"{self.model_type}_confusion.png")


# ---------------------------------------------------------------------- I/O
def load_data(expr_path, labels_path):
    expr = pd.read_csv(expr_path, index_col=0)
    expr.columns = expr.columns.str.split(".").str[0]
    labels = pd.read_csv(labels_path, index_col=0)
    labels.index.name = "SampleID"
    labels.index = labels.index.astype(str).str.strip().str.upper()
    expr.index = expr.index.astype(str).str.strip().str.upper()
    common = expr.index.intersection(labels.index)
    expr, labels = expr.loc[common], labels.loc[common]
    print(f"Aligned expression matrix: {expr.shape} | labels: {labels.shape}")
    return expr, labels


def save_model_info(model, path, n_samples):
    ext = "cbm" if model.model_type == "catboost" else "json"
    with open(path, "w") as f:
        f.write(
            f"# Diagnostic Model Summary ({model.model_type.upper()})\n\n"
            f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"Test AUC: {model.test_auc:.3f}\n"
            f"CV AUC: {model.cv_scores.mean():.3f} ± {model.cv_scores.std():.3f}\n"
            f"Samples: {n_samples}\n"
            f"Best params: {model.best_params}\n"
            f"Genes ({len(model.available_genes)}): {', '.join(model.available_genes)}\n\n"
            f"Files: {model.model_type}_predictions.csv, {model.model_type}_model.{ext}\n"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expression", default="data/expression_matrix.csv")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", nargs="+", default=["catboost", "xgboost"],
                    choices=["catboost", "xgboost"])
    ap.add_argument("--n-iter", type=int, default=60, help="Bayesian search iterations")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="Parallel jobs for the search (-1 = all cores; use 1 if workers crash)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-to-train", default="",
                    help="Sample ID to force into the training set (optional)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    expr, labels = load_data(args.expression, args.labels)

    summary = {}
    for mt in args.models:
        m = DiagnosticModel(expr, labels, model_type=mt, n_iter=args.n_iter,
                            seed=args.seed, outdir=args.outdir, n_jobs=args.n_jobs)
        m.run_full_pipeline(tune=not args.no_tune, sample_to_train=args.sample_to_train.upper())
        m.classify_new_samples(expr).to_csv(
            os.path.join(args.outdir, f"{mt}_predictions.csv"), index=False)
        m.model.save_model(os.path.join(
            args.outdir, f"{mt}_model.{'cbm' if mt == 'catboost' else 'json'}"))
        save_model_info(m, os.path.join(args.outdir, f"{mt}_model_info.txt"), len(expr))
        summary[mt] = m.test_auc

    print("\n=== Test AUC comparison ===")
    for k, v in summary.items():
        print(f"{k:10s} {v:.3f}")


if __name__ == "__main__":
    main()
