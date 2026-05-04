# =============================================================================
# preprocessing.R
# Stratified train/test split, recipes preprocessing pipeline, SMOTE setup
#
# Requires: dr_raw (from 01_load_and_eda.R)
# Outputs : dr_train, dr_test, dr_recipe, dr_cv_folds
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(themis)       # SMOTE step for recipes
})

set.seed(42)  # reproducibility

# ── 1. Stratified train / test split (80/20) ─────────────────────────────────
# Stratification ensures both splits reflect the original class ratio.
# This is critical — a random split on small datasets can inadvertently
# create unbalanced subsets.

dr_split <- initial_split(dr_raw, prop = 0.80, strata = label)
dr_train <- training(dr_split)
dr_test  <- testing(dr_split)

cat("── Train/Test Split ──\n")
cat(sprintf("  Training: %d rows | Positive rate: %.1f%%\n",
            nrow(dr_train),
            mean(dr_train$label == "DR Present") * 100))
cat(sprintf("  Testing:  %d rows | Positive rate: %.1f%%\n",
            nrow(dr_test),
            mean(dr_test$label == "DR Present") * 100))

# ── 2. Cross-validation folds (for model tuning) ─────────────────────────────
# 5-fold CV on training set — stratified.
# SMOTE will be applied *inside* each fold during workflow fitting,
# preventing synthetic samples from leaking into validation folds.

dr_cv_folds <- vfold_cv(dr_train, v = 5, strata = label)
cat("\n5-fold CV folds created (stratified)\n")

# ── 3. Preprocessing recipe ───────────────────────────────────────────────────
# A tidymodels recipe is a reproducible, leakage-safe preprocessing pipeline.
# All transformations are *estimated* on training data only, then applied
# to test data — no information from the test set bleeds in.

dr_recipe <- recipe(label ~ ., data = dr_train) %>%

  # Step 1: Convert binary factors to dummy variables
  # (quality, pre_screening, amfm_class are 0/1 but encoded as factors)
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%

  # Step 2: Remove near-zero variance predictors
  # Some MA detection columns at very high confidence thresholds may have
  # almost no variance (most patients = 0). These add noise, not signal.
  step_nzv(all_predictors()) %>%

  # Step 3: Normalize all numeric predictors to [0, 1] range
  # Required for logistic regression and SVM. Doesn't hurt RF/XGBoost
  # but keeps the recipe universal across models.
  step_range(all_numeric_predictors()) %>%

  # Step 4: SMOTE — Synthetic Minority Oversampling Technique
  # Generates synthetic positive (DR) examples by interpolating between
  # existing minority class neighbors in feature space.
  #
  # CRITICAL DESIGN NOTE: SMOTE is applied *inside* the recipe, which
  # means it only fires during training (not on test/validation data).
  # When used inside a workflow with CV, it runs inside each fold,
  # preventing the common error of generating synthetic samples before
  # the train/validation split.
  #
  # over_ratio = 0.9 → brings minority class to 90% of majority class size.
  # Full 1:1 balance can sometimes hurt if the dataset is already close
  # to balanced; 0.9 is a reasonable starting point.
  step_smote(label, over_ratio = 0.9, seed = 42)

# ── 4. Inspect the recipe (doesn't fit yet) ──────────────────────────────────

cat("\n── Recipe Summary ──\n")
print(dr_recipe)

# Prep and bake on training data to show what preprocessing does
prepped <- prep(dr_recipe, training = dr_train)
baked   <- bake(prepped, new_data = NULL)

cat("\nAfter preprocessing — Training set dimensions:\n")
cat(sprintf("  Rows (with SMOTE): %d | Columns: %d\n",
            nrow(baked), ncol(baked)))
cat("\nClass distribution after SMOTE:\n")
print(table(baked$label))

cat("\n── Preprocessing complete — objects available: ──\n")
cat("  dr_train, dr_test, dr_recipe, dr_cv_folds\n")
