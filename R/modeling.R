# =============================================================================
# modeling.R
# Define, tune, and fit four classifiers using tidymodels workflows
#
# Requires: dr_train, dr_recipe, dr_cv_folds (from 02_preprocessing.R)
# Outputs : best_lr, best_rf, best_xgb, best_svm (fitted workflow objects)
#           model_cv_results (CV comparison table)
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(ranger)       # fast random forest backend
  library(xgboost)      # gradient boosting backend
  library(kernlab)      # SVM backend
  library(doParallel)   # parallel CV (optional but helpful)
})

# Enable parallel processing for CV tuning (uses all available cores - 1)
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
cat(sprintf("Parallel processing: %d cores\n\n", n_cores))

# Metric set for model selection — we care about AUROC primarily,
# but track sensitivity and specificity explicitly for clinical framing
cv_metrics <- metric_set(roc_auc, sensitivity, specificity, f_meas, pr_auc)

# =============================================================================
# MODEL 1: Logistic Regression (L2 regularized via glmnet)
# =============================================================================
# Clinical role: interpretable baseline. Log-odds coefficients have direct
# clinical meaning. Regularization (penalty) handles correlated features.

cat("── Model 1: Logistic Regression ──\n")

lr_spec <- logistic_reg(
  penalty = tune(),   # L2 regularization strength (lambda)
  mixture = 0         # mixture = 0 → Ridge (L2); 1 → Lasso (L1)
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

lr_workflow <- workflow() %>%
  add_recipe(dr_recipe) %>%
  add_model(lr_spec)

lr_grid <- grid_regular(
  penalty(range = c(-4, 0), trans = log10_trans()),
  levels = 20
)

lr_tune <- tune_grid(
  lr_workflow,
  resamples = dr_cv_folds,
  grid      = lr_grid,
  metrics   = cv_metrics,
  control   = control_grid(save_pred = TRUE, verbose = FALSE)
)

# Select best model by AUROC
lr_best_params <- select_best(lr_tune, metric = "roc_auc")
best_lr <- lr_workflow %>%
  finalize_workflow(lr_best_params) %>%
  fit(dr_train)

cat(sprintf("  Best penalty: %.5f | CV AUROC: %.4f\n",
            lr_best_params$penalty,
            show_best(lr_tune, metric = "roc_auc", n = 1)$mean))

# =============================================================================
# MODEL 2: Random Forest
# =============================================================================
# Clinical role: strong ensemble; handles feature correlations well (common
# in retinal feature datasets). Native variable importance is interpretable.

cat("\n── Model 2: Random Forest ──\n")

rf_spec <- rand_forest(
  mtry  = tune(),   # features considered at each split
  trees = 500,      # enough trees for stable estimates
  min_n = tune()    # minimum node size (controls tree depth)
) %>%
  set_engine("ranger",
             importance  = "impurity",   # Gini importance saved for VIP
             num.threads = n_cores) %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(dr_recipe) %>%
  add_model(rf_spec)

# finalize mtry range based on actual number of predictors
rf_params <- rf_workflow %>%
  extract_parameter_set_dials() %>%
  finalize(dr_train %>% select(-label))

rf_grid <- grid_latin_hypercube(rf_params, size = 25)

rf_tune <- tune_grid(
  rf_workflow,
  resamples = dr_cv_folds,
  grid      = rf_grid,
  metrics   = cv_metrics,
  control   = control_grid(save_pred = TRUE, verbose = FALSE)
)

rf_best_params <- select_best(rf_tune, metric = "roc_auc")
best_rf <- rf_workflow %>%
  finalize_workflow(rf_best_params) %>%
  fit(dr_train)

cat(sprintf("  Best mtry: %d | min_n: %d | CV AUROC: %.4f\n",
            rf_best_params$mtry,
            rf_best_params$min_n,
            show_best(rf_tune, metric = "roc_auc", n = 1)$mean))

# =============================================================================
# MODEL 3: XGBoost (Gradient Boosting)
# =============================================================================
# Clinical role: typically highest performance on tabular data. Tree-based
# boosting corrects sequential errors. More hyperparameters to tune.

cat("\n── Model 3: XGBoost ──\n")

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) %>%
  set_engine("xgboost",
             nthread     = n_cores,
             eval_metric = "auc") %>%
  set_mode("classification")

xgb_workflow <- workflow() %>%
  add_recipe(dr_recipe) %>%
  add_model(xgb_spec)

xgb_params <- xgb_workflow %>%
  extract_parameter_set_dials() %>%
  finalize(dr_train %>% select(-label))

# Latin hypercube gives good coverage of a 6-dimensional parameter space
xgb_grid <- grid_latin_hypercube(xgb_params, size = 40)

xgb_tune <- tune_grid(
  xgb_workflow,
  resamples = dr_cv_folds,
  grid      = xgb_grid,
  metrics   = cv_metrics,
  control   = control_grid(save_pred = TRUE, verbose = FALSE)
)

xgb_best_params <- select_best(xgb_tune, metric = "roc_auc")
best_xgb <- xgb_workflow %>%
  finalize_workflow(xgb_best_params) %>%
  fit(dr_train)

cat(sprintf("  Best trees: %d | depth: %d | lr: %.4f | CV AUROC: %.4f\n",
            xgb_best_params$trees,
            xgb_best_params$tree_depth,
            xgb_best_params$learn_rate,
            show_best(xgb_tune, metric = "roc_auc", n = 1)$mean))

# =============================================================================
# MODEL 4: Support Vector Machine (Radial Kernel)
# =============================================================================
# Clinical role: strong on small clinical datasets; effective when classes
# are not linearly separable in feature space.

cat("\n── Model 4: SVM (Radial Kernel) ──\n")

svm_spec <- svm_rbf(
  cost      = tune(),   # misclassification penalty
  rbf_sigma = tune()    # kernel bandwidth
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_workflow <- workflow() %>%
  add_recipe(dr_recipe) %>%
  add_model(svm_spec)

svm_grid <- grid_regular(
  cost(range      = c(-2, 4)),
  rbf_sigma(range = c(-4, 0)),
  levels = 8
)

svm_tune <- tune_grid(
  svm_workflow,
  resamples = dr_cv_folds,
  grid      = svm_grid,
  metrics   = cv_metrics,
  control   = control_grid(save_pred = TRUE, verbose = FALSE)
)

svm_best_params <- select_best(svm_tune, metric = "roc_auc")
best_svm <- svm_workflow %>%
  finalize_workflow(svm_best_params) %>%
  fit(dr_train)

cat(sprintf("  Best cost: %.3f | sigma: %.5f | CV AUROC: %.4f\n",
            svm_best_params$cost,
            svm_best_params$rbf_sigma,
            show_best(svm_tune, metric = "roc_auc", n = 1)$mean))

# ── Stop parallel cluster ─────────────────────────────────────────────────────
stopCluster(cl)

# =============================================================================
# Cross-validation comparison table
# =============================================================================

collect_cv_metric <- function(tune_result, model_name) {
  collect_metrics(tune_result) %>%
    filter(.metric %in% c("roc_auc", "sensitivity", "specificity")) %>%
    group_by(.metric) %>%
    slice_max(mean, n = 1) %>%
    ungroup() %>%
    select(.metric, mean, std_err) %>%
    mutate(model = model_name)
}

model_cv_results <- bind_rows(
  collect_cv_metric(lr_tune,  "Logistic Regression"),
  collect_cv_metric(rf_tune,  "Random Forest"),
  collect_cv_metric(xgb_tune, "XGBoost"),
  collect_cv_metric(svm_tune, "SVM")
)

cat("\n── CV Comparison (best hyperparameters per model) ──\n")
model_cv_results %>%
  pivot_wider(names_from = .metric, values_from = c(mean, std_err)) %>%
  select(model, mean_roc_auc, mean_sensitivity, mean_specificity) %>%
  mutate(across(where(is.numeric), ~round(.x, 4))) %>%
  print()

# =============================================================================
# Save fitted models
# =============================================================================

saveRDS(best_lr,  "outputs/models/best_lr.rds")
saveRDS(best_rf,  "outputs/models/best_rf.rds")
saveRDS(best_xgb, "outputs/models/best_xgb.rds")
saveRDS(best_svm, "outputs/models/best_svm.rds")
saveRDS(model_cv_results, "outputs/models/cv_results.rds")

cat("\nModels saved to outputs/models/\n")
cat("Objects available: best_lr, best_rf, best_xgb, best_svm, model_cv_results\n")
