# Diabetic Retinopathy Detection: An End-to-End ML Pipeline in R

[![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r)](https://www.r-project.org/) [![tidymodels](https://img.shields.io/badge/tidymodels-1.1%2B-orange)](https://www.tidymodels.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Clinical Problem

Diabetic retinopathy (DR) is the leading cause of preventable blindness in working-age adults. Approximately 1 in 3 people with diabetes show signs of DR, yet access to ophthalmology is limited in many settings. Automated screening using fundus image analysis can flag high-risk patients for referral, but the *cost of a false negative* (missed DR) is vision loss, while a false positive means an unnecessary clinic visit. This asymmetry drives every modeling decision in this project.

This pipeline builds and evaluates machine learning classifiers on clinically-extracted retinal image features, with explicit sensitivity/specificity tradeoff analysis and SHAP-based model interpretability which are the kinds of outputs required for real clinical deployment.

> **Connection to genomic ML:** The same modeling challenges appear in CRISPR off-target prediction - rare positive events, severe cost asymmetry between false negatives and false positives, and the need for interpretable predictions that can be reviewed by domain experts. This project demonstrates those skills in a well-characterized clinical dataset.

------------------------------------------------------------------------

## Dataset

**Source:** [UCI Machine Learning Repository — Diabetic Retinopathy Debrecen](https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen+data+set)

**Reference:** Antal, B., & Hajdu, A. (2014). An ensemble-based system for automatic screening of diabetic retinopathy. *Knowledge-Based Systems*, 60, 20–27.

### Feature Glossary

| \# | Feature | Clinical Meaning |
|----|----|----|
| 1 | `quality` | Binary quality assessment of the fundus image (0 = failed, 1 = sufficient) |
| 2 | `pre_screening` | Binary pre-screening result for severe retinal abnormality |
| 3–8 | `ma_detect_0.5`–`ma_detect_1.0` | Microaneurysm (MA) detections at confidence levels 0.5–1.0. MAs are early hallmarks of DR — small bulges in retinal capillaries |
| 9–16 | `exudate_1`–`exudate_8` | Normalized exudate (lipid deposit) features. Hard exudates are a sign of vascular leakage |
| 17 | `macula_od_dist` | Euclidean distance from the macula to the optic disc, normalized to image diameter |
| 18 | `optic_disc_diam` | Diameter of the optic disc |
| 19 | `amfm_class` | AM/FM-based classification of retinal images (pre-classifier feature) |
| 20 | `label` | **Target**: 1 = signs of DR present, 0 = no DR |

------------------------------------------------------------------------

## Methods Overview

```         
Raw Features → EDA → Preprocessing (SMOTE inside CV) → Model Training → Clinical Evaluation → SHAP Interpretability
```

### Models Compared

-   **Logistic Regression** (L2 regularized) — clinical baseline; interpretable log-odds
-   **Random Forest** (`ranger`) — strong ensemble; handles feature correlations
-   **XGBoost** — state-of-the-art tabular performance; hyperparameter tuned
-   **SVM** (radial kernel) — included for breadth; effective on small clinical datasets

### Key Design Decisions

-   **SMOTE applied inside CV folds only** — prevents synthetic sample leakage into validation sets
-   **Threshold optimization** — models evaluated at default (0.5) and sensitivity-optimized thresholds (≥90% sensitivity)
-   **AUROC + Precision-Recall** — both reported; PR curves are more informative under class imbalance
-   **Calibration** — predicted probabilities checked against observed rates (critical for clinical deployment)

------------------------------------------------------------------------

## Key Results

> *(Regenerate by running the pipeline — see below)*

| Model               | AUROC | Sensitivity | Specificity | F1  |
|---------------------|-------|-------------|-------------|-----|
| Logistic Regression | —     | —           | —           | —   |
| Random Forest       | —     | —           | —           | —   |
| XGBoost             | —     | —           | —           | —   |
| SVM                 | —     | —           | —           | —   |

*Table populated after running `05_run_all.R`*

------------------------------------------------------------------------

## Repository Structure

```         
diabetic-retinopathy-ml/
├── README.md
├── data/
│   └── raw/                        ← download dataset here (see Setup)
├── R/
│   ├── load_and_eda.R           ← data loading, feature glossary, EDA plots
│   ├── preprocessing.R          ← recipes pipeline, SMOTE, train/test split
│   ├── modeling.R               ← tidymodels workflows, CV tuning, model fits
│   ├── evaluation.R             ← ROC, PR curves, confusion matrices, calibration
│   ├── interpretability.R       ← SHAP values, VIP plots, PDP, case narratives
│   └── run_all.R                ← sources all scripts in order
├── reports/
│   └── analysis_report.Rmd         ← knits to final HTML report
├── outputs/
│   ├── figures/                    ← all saved plots (PNG)
│   └── models/                     ← saved model objects (RDS)
├── renv.lock                       ← exact package versions for reproducibility
└── .gitignore
```

------------------------------------------------------------------------

## Setup & Reproducibility

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy-ml.git
cd diabetic-retinopathy-ml
```

### 2. Download the dataset

Go to: <https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen+data+set>

Download `messidor_features.arff` and place it at:

```         
data/raw/messidor_features.arff
```

### 3. Restore the R environment

``` r
install.packages("renv")
renv::restore()
```

### 4. Run the full pipeline

``` r
source("R/run_all.R")
```

### 5. Generate the report

``` r
rmarkdown::render("reports/analysis_report.Rmd", output_dir = "outputs/")
```

------------------------------------------------------------------------

## Limitations

-   **Feature extraction pipeline is fixed:** This dataset uses pre-extracted features from a specific image analysis system. Model performance may not generalize to features extracted by different fundus camera software.
-   **Single-site data:** Images collected at one center (Debrecen, Hungary). Performance on different camera types, lighting conditions, or ethnic populations is unknown.
-   **ARFF format features:** The 8 exudate features are already normalized — raw spatial information is lost.
-   **No temporal data:** DR is a progressive disease; a single-timepoint classifier cannot capture longitudinal change.

------------------------------------------------------------------------

## Author

**Cynric Joshua Muller**\
MS Bioinformatics\
[LinkedIn] \| [GitHub]

------------------------------------------------------------------------

## License

MIT License — see [LICENSE](LICENSE) for details.
