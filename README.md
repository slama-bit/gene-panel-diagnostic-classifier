# Gene-Panel Diagnostic Classifier: HCC vs Non-Tumor Liver

This project classifies **hepatocellular carcinoma (HCC) vs non-tumor liver** from microarray expression data. It uses a curated **29-gene panel** covering EMT, apoptosis, glycolysis/OXPHOS, lactate transport, proliferation and inflammation. Two gradient-boosting models, **CatBoost** and **XGBoost**, are tuned with Bayesian optimisation and interpreted with **SHAP**. Every result is evaluated with **patient-grouped** splits, so replicate arrays from the same patient never appear in both training and test sets.

**Dataset:** [GEO GSE62232](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62232), Affymetrix HG-U133 Plus 2.0. It has 81 HCC arrays (label 1) and 10 non-tumor liver arrays (label 0). The 10 non-tumor arrays come from **5 patients only**.

## Pipeline

1. **Load and align**: the expression matrix (samples × genes) and the labels are matched on normalised sample IDs.
2. **Feature selection**: the matrix is restricted to the predefined gene panel. The genes come from prior knowledge, not from the data.
3. **Split**: a stratified **group** split (`--group-pattern`) keeps all arrays from one patient on one side of the split.
4. **Tune**: `BayesSearchCV` with grouped 5-fold CV and ROC-AUC scoring, run on the training set only.
5. **Evaluate**: classification report, ROC, precision-recall and confusion matrix on the held-out patients, plus grouped 5-fold CV AUC.
6. **Explain**: SHAP mean |value| ranking.
7. **Robustness**: `src/robustness_checks.py` measures replicate leakage and compares the panel against random gene panels.

### Gene panel

| Module | Genes |
|---|---|
| EMT | CDH1, CDH2, SNAI1, SNAI2, VIM, ACTA2 |
| Apoptosis / p53 | TP53, BAX, BCL2, CASP3, CDKN1A |
| Mitochondrial / OXPHOS | COX11, ATP6V1H, NDUFAF5, NDUFA10, ESRRA |
| Glycolysis | PKM, TPI1, LDHA, ENO1 |
| Lactate transport | SLC16A1, SLC16A4 |
| Proliferation | MKI67, TERT, TOP1, MYC |
| Signalling / inflammation | FGFR1, CCL2, CXCL8 |

## Results

These results come from a patient-grouped split with 60 Bayesian tuning iterations. The test set holds 28 patients (30 arrays): 27 HCC arrays and 3 non-tumor arrays. All 3 non-tumor arrays come from **one** patient (0934N).

| Model | Test AUC | Non-tumor recall @0.5 | HCC recall | Grouped 5-fold CV AUC |
|---|---|---|---|---|
| CatBoost | 1.000 | 0.67 (2/3) | 1.00 | 1.000 ± 0.000 |
| XGBoost  | 0.951 | 0.00 (0/3) | 1.00 | 0.996 ± 0.008 |

**Top genes in both models:** BAX, SLC16A4 and SNAI1, followed by glycolytic genes (PKM, TPI1, ENO1).

The ranking is good, but the default 0.5 threshold misclassifies non-tumor tissue. XGBoost scores every array of the held-out non-tumor patient above 0.5. This comes from the 81:10 class imbalance, so the threshold and class weights need calibrating.

| | CatBoost | XGBoost |
|---|---|---|
| ROC | ![](results/figures/catboost_roc.png) | ![](results/figures/xgboost_roc.png) |
| Confusion | ![](results/figures/catboost_confusion.png) | ![](results/figures/xgboost_confusion.png) |
| SHAP | ![](results/figures/catboost_shap.png) | ![](results/figures/xgboost_shap.png) |

Exploratory plots: [class distribution](results/figures/class_distribution.png) · [PCA variance](results/figures/pca_variance.png)

### Robustness checks

Full output is in [`results/robustness_summary.txt`](results/robustness_summary.txt).

| Check | Result |
|---|---|
| Independent non-tumor patients | **5** (10 arrays: 3 + 3 + 2 + 1 + 1) |
| Original Colab split: non-tumor test arrays whose patient was also in training | **3 / 3** (leakage) |
| Curated panel, grouped CV AUC | 1.000 |
| **Random 29-gene panels** (n = 30), grouped CV AUC | mean 0.981, median 0.992; 23 % reach 1.000 |
| Single genes, grouped CV AUC | SLC16A4 0.953 · BAX 0.940 · SNAI1 0.894 · PKM 0.853 · ENO1 0.814 |

**What these checks show.** HCC and non-tumor liver are very easy to separate with this data: random gene sets reach near-perfect AUC. The curated panel works, but these results do **not** show that it is specifically informative. AUC alone cannot tell it apart from random genes on this dataset.

## Limitations

- **There are only 5 independent non-tumor patients.** Every minority-class estimate rests on 1–2 patients per fold. Sample-level metrics overstate the effective sample size.
- **The original Colab run (now archived) leaked replicates.** Arrays from the same patient sat in both train and test, and one replicate (GSM1523338, 0934N) was forced into training. Those figures are kept in [`results/figures/original_colab_run/`](results/figures/original_colab_run/) for transparency only.
- **The default threshold is wrong for 81:10 imbalance.** Use class weights (`auto_class_weights="Balanced"`, `scale_pos_weight`) or choose a threshold with grouped CV.
- **CV after tuning is mildly optimistic.** Nested grouped CV would give an unbiased estimate.
- **There is no external cohort.** The next step is validation on an independent HCC dataset, such as TCGA-LIHC or another GEO series. A harder clinical question, such as HCC vs cirrhosis or early vs late stage, would test whether the panel adds value over random genes.

## Usage

```bash
git clone https://github.com/slama-bit/gene-panel-diagnostic-classifier.git
cd gene-panel-diagnostic-classifier
pip install -r requirements.txt

# put expression_matrix.csv and labels.csv in data/ (see data/README.md)

# main pipeline, leakage-free
python src/diagnostic_model.py --group-pattern 'JZN_(\d+[NT])' --n-iter 60

# robustness checks
python src/robustness_checks.py --n-random 30
```

| Flag | Default | Meaning |
|---|---|---|
| `--group-pattern` | none | Regex whose first capture group defines the patient. Enables grouped split and CV. **Recommended.** |
| `--models` | `catboost xgboost` | Which models to run |
| `--n-iter` | 60 | Bayesian search iterations |
| `--no-tune` | off | Skip tuning |
| `--n-jobs` | -1 | Parallel workers for the search. Use `1` if workers crash. |
| `--sample-to-train` | none | Force a sample into training (ungrouped mode only; kept to reproduce the original run) |
| `--seed` | 42 | Random seed |

The original Colab analysis, with its outputs, is in [`notebooks/diagnostic_model.ipynb`](notebooks/diagnostic_model.ipynb).

## Repository layout

```
├── src/
│   ├── diagnostic_model.py       # pipeline (CLI)
│   └── robustness_checks.py      # leakage + random-panel baseline
├── notebooks/                    # original Colab notebook
├── results/
│   ├── robustness_summary.txt
│   ├── *_grouped_run_info.txt
│   └── figures/                  # grouped run; original_colab_run/ archived
├── data/                         # input data (not tracked, see data/README.md)
└── requirements.txt
```

## License

MIT
