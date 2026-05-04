# =============================================================================
# 00_run_all.R
# Master script — sources all pipeline stages in order
#
# Usage: source("R/00_run_all.R")
# Or from terminal: Rscript R/00_run_all.R
#
# Prerequisites:
#   1. renv::restore() to install all packages
#   2. data/raw/messidor_features.arff downloaded from UCI
# =============================================================================

cat("\n", strrep("=", 60), "\n")
cat("  Diabetic Retinopathy ML Pipeline\n")
cat("  Starting:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(strrep("=", 60), "\n\n")

# Create output directories if they don't exist
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/models",  recursive = TRUE, showWarnings = FALSE)

pipeline_start <- proc.time()

# ── Stage 1: EDA ─────────────────────────────────────────────────────────────
cat("── Stage 1/5: Exploratory Data Analysis ──\n")
source("R/load_and_eda.R")
cat("   ✓ EDA complete\n\n")

# ── Stage 2: Preprocessing ───────────────────────────────────────────────────
cat("── Stage 2/5: Preprocessing ──\n")
source("R/preprocessing.R")
cat("   ✓ Preprocessing complete\n\n")

# ── Stage 3: Modeling ────────────────────────────────────────────────────────
cat("── Stage 3/5: Model Training & Tuning ──\n")
cat("   Note: CV tuning may take several minutes...\n")
source("R/modeling.R")
cat("   ✓ Modeling complete\n\n")

# ── Stage 4: Evaluation ──────────────────────────────────────────────────────
cat("── Stage 4/5: Clinical Evaluation ──\n")
source("R/evaluation.R")
cat("   ✓ Evaluation complete\n\n")

# ── Stage 5: Interpretability ────────────────────────────────────────────────
cat("── Stage 5/5: Interpretability (SHAP) ──\n")
cat("   Note: SHAP computation may take a few minutes...\n")
source("R/interpretability.R")
cat("   ✓ Interpretability complete\n\n")

# ── Summary ──────────────────────────────────────────────────────────────────
elapsed <- proc.time() - pipeline_start
cat(strrep("=", 60), "\n")
cat("  Pipeline complete!\n")
cat(sprintf("  Total time: %.1f seconds\n", elapsed["elapsed"]))
cat("  Outputs saved to: outputs/figures/ and outputs/models/\n")
cat("  Next: rmarkdown::render('reports/analysis_report.Rmd')\n")
cat(strrep("=", 60), "\n\n")
