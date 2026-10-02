# Early-Stage ASD Screening from AQ-10 Data

Machine learning evaluation for autism spectrum disorder (ASD) screening on the Kaggle *Autism Prediction* dataset. Ten classifiers and two fusion models are compared against the standard AQ-10 cutoff (score ≥ 6) with repeated stratified cross-validation.

The notebook reimplements and extends the workflow of the MSc thesis *Early-stage ASD Detection from Clinical Test Data* (Jahangirnagar University, 2024).

## Key findings

- **The models match the AQ-10 rule but do not clearly beat it.** The AQ sum alone reaches an AUC of 0.91, and no model exceeds it beyond noise.
- **Accuracy is misleading.** Only 20% of the records are positive, so models look better at the default threshold mainly because they flag fewer people and miss more real cases.
- **At equal recall, the gain is small.** At a matched recall of about 90%, the best models (fusion, K-Means, Fuzzy C-Means) raise specificity by roughly 4–5 points over an AQ cutoff chosen on the training data. About half of the flagged records are still negative.
- **Fusion adds little.** Entropy weighting performs no better than an equal-weight average.
- **The `result` column adds almost nothing.** Results with and without it are nearly identical.

### Cross-validated results at matched recall (without `result`)

Mean over 10 folds (5-fold, 2 repeats). Thresholds are chosen on out-of-fold training predictions.

| Model | Recall | Specificity | Precision | AUC |
|---|---|---|---|---|
| AQ sum (cutoff chosen on train) | 0.913 | 0.743 | 0.474 | 0.911 |
| Fusion, equal-weight | 0.897 | 0.788 | 0.520 | 0.908 |
| Fusion, entropy-weighted | 0.904 | 0.782 | 0.516 | 0.912 |
| K-Means | 0.885 | 0.779 | 0.505 | 0.901 |
| Fuzzy C-Means | 0.891 | 0.778 | 0.506 | 0.897 |
| MLR + Logistic Regression | 0.897 | 0.761 | 0.490 | 0.895 |
| AdaBoost | 0.910 | 0.754 | 0.483 | 0.897 |
| KNN | 0.919 | 0.751 | 0.484 | 0.890 |
| LDA | 0.901 | 0.750 | 0.480 | 0.895 |
| Naive Bayes | 0.910 | 0.750 | 0.482 | 0.891 |
| Random Forest | 0.904 | 0.744 | 0.473 | 0.895 |

The decision tree (AUC 0.85) and the SVM on the MLR score and age (AUC 0.84) are unstable at a recall target, with specificity varying strongly between folds. Per-fold standard deviations, the default-threshold results and the paired comparisons against the AQ baseline are in section 9 of the notebook. Differences of about one standard deviation (roughly 0.03 in specificity) should be treated as noise.

## Method

1. **Cleaning.** Deterministic, row-wise mappings only. All imputation, scaling and encoding run inside scikit-learn pipelines and are fitted on training folds only.
2. **Models.** AdaBoost, decision tree, random forest, Gaussian Naive Bayes, LDA, KNN, Fuzzy C-Means, K-Means, multiple linear regression (MLR) with logistic regression, and an RBF SVM on the MLR score and age. A supplementary SVM on age and `result` is reported separately. Hyperparameters are fixed, not tuned.
3. **Decision tree size.** Chosen with the one standard error rule inside each training fold.
4. **Clustering classifiers.** Each of the two clusters takes the class with the higher positive rate, so both classes are always predicted.
5. **Calibration and fusion.** Platt sigmoids are fitted on out-of-fold training probabilities. Entropy-weighted fusion weights each probability by `1 - H(p)`, with an equal-weight average as a control.
6. **Evaluation.** Repeated stratified cross-validation (5 folds, 2 repeats), run with and without `result`. Each model is scored at the default threshold (`p ≥ 0.5`) and at a matched recall (`TARGET_RECALL = 0.90`). Baselines are the AQ-10 ≥ 6 rule, an AQ-sum cutoff chosen on the training data, and the majority class. Per-fold paired differences against the baseline are descriptive only, because the folds overlap and p-values would be invalid.

## Repository structure

```
.
├── ASD_detection_extended.ipynb   # full pipeline with saved outputs
└── README.md
```

`train.csv` is not included. Data is not redistributed here.

## Getting started

1. Download `train.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/autismdiagnosis) (a Kaggle account may be required) and place it next to the notebook or in `./data/`.
2. Install the dependencies (Python 3.9+):
   ```bash
   pip install numpy pandas matplotlib seaborn "scikit-learn>=1.2"
   ```
3. Open the notebook and run all cells. A full run takes a few minutes.

In Colab, the notebook prompts for the file if it is not found. In a Kaggle notebook it looks in `/kaggle/input/`. The loader stops with an error if the file is missing or lacks the expected columns, so the UCI version of the dataset is not supported.

### Configuration

Settings are in the first code cell:

| Setting | Meaning |
|---|---|
| `USE_RESULT` | Include the `result` column in the hold-out run. Section 9 reports both settings |
| `TARGET_RECALL` | Recall for the matched-recall operating point (default 0.90) |
| `N_SPLITS`, `N_REPEATS` | Cross-validation folds and repeats (default 5 and 2) |
| `PARAMS` | Fixed model hyperparameters |
| `RANDOM_STATE` | Seed (default 42) |

## Limitations

- The dataset is small (800 records), and some columns look artificial: `result` is not the AQ sum and contains negative values.
- Results come from one dataset and need checking on real clinical data.
- Gaps among the best models are within fold-to-fold variation, so their ranking is not reliable.
- The models are not diagnostic tools and do not replace a clinical assessment.

## Data and credits

Dataset: Kaggle *Autism Prediction in Adults* (REVA Academy). Please follow the competition's terms of use. The AQ-10 screening instrument is by Allison, Auyeung and Baron-Cohen.
