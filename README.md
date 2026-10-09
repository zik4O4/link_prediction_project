# Graph Link Prediction Experiments

A Python experiment pipeline comparing network similarity features and supervised classifiers across Facebook, dolphins, football and US power-grid graphs.

## Problem and solution

The project studies how graph-based features influence binary edge classification. It calculates ten classical similarity features plus `Our_measure`, compares models with and without that feature, and exports tables and plots.

**Important evaluation limitation:** features are calculated on the original graph before train/test row splitting. Positive test edges are still present in that graph, and the custom feature counts direct paths. Existing scores therefore do not establish leakage-free prediction of unseen missing links. A proper edge-holdout experiment is future work; existing results are preserved.

## Architecture

| Module | Responsibility |
|---|---|
| `src/data_loader.py` | Load edge-list/Matrix Market graphs and keep the largest connected component |
| `src/similarity_metrics.py` | Compute graph similarity features |
| `src/feature_extraction.py` | Sample positive/negative pairs, extract features and split rows |
| `src/train_models.py` | Compare KNN, logistic regression, random forest, MLP and SVM; optional XGBoost |
| `src/visualization.py` | Export figures and results tables |
| `src/main.py` | Run the experiment pipeline |

**Stack:** Python, NetworkX, NumPy, pandas, SciPy, scikit-learn, Matplotlib and Seaborn.

## Installation and execution

```bash
git clone https://github.com/zik4O4/link_prediction_project.git
cd link_prediction_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.main
```

Run from the repository root because dataset/output paths are relative to it. Execution can be expensive: graph diameter and bounded simple-path enumeration may take substantial time. A new run writes `results/figures/results_comparison.png` and `results/tables/results_table.csv`; copy existing artifacts locally first if you need to retain them unchanged.

`tqdm` is optional. Installing `xgboost` adds another model to the default comparison; this changes the set of experiments. Dependencies have lower bounds rather than a locked environment.

## Data and saved outputs

The four included datasets are under `datasets/facebook`, `datasets/dolphins`, `datasets/football` and `datasets/power`. Per-dataset `readme.html` files contain source information; verify upstream attribution and terms before redistributing. The download helper is not required for the included files and its dataset list is not identical to the main loader.

Existing outputs:

- [`results/tables/results_table.csv`](results/tables/results_table.csv): accuracy comparisons with and without `Our_measure`.
- [`results/figures/`](results/figures/): comparison and feature-importance plots.
- [`models/`](models/): historical fitted model artifacts and performance plots.
- [`datasets/features/`](datasets/features/): previously exported feature matrices and labels.

The saved comparisons show both positive and negative gains depending on graph/model. They are historical artifacts, not newly reproduced benchmarks. Only load pickle files from sources you trust and with a compatible dependency environment.

## Methodology limitations

- Edge existence leaks into features under the current evaluation protocol.
- Negative sampling and NumPy shuffling do not establish a reproducible global seed.
- Scaling is fitted before cross-validation fold splitting, so CV estimates need a fold-aware pipeline in a future experiment.
- `Our_measure` uses a path-length bound and count cap; its interpretation and computational cost should be documented before claiming generality.
- Scientific definitions, training logic and stored results are unchanged by the documentation/import fixes.

## License

No repository-level license has been selected. Dataset provenance and third-party terms must be retained separately.
