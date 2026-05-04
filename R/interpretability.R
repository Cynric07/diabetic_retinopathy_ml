# =============================================================================
# interpretability.R
# Model interpretability: SHAP values, variable importance, PDPs,
# and individual case narratives for clinical explainability
#
# Requires: best_rf, best_xgb, dr_train, dr_test (from prior scripts)
# Outputs : outputs/figures/interp_*.png
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(vip)          # variable importance plots (model-agnostic API)
  library(shapviz)      # SHAP visualization
  library(kernelshap)   # model-agnostic SHAP computation
  library(patchwork)
  library(ggbeeswarm)   # beeswarm plots for SHAP summary
})

# Prepare preprocessed training data (needed for SHAP background)
prepped_recipe <- prep(dr_recipe, training = dr_train)
train_baked    <- bake(prepped_recipe, new_data = dr_train)
test_baked     <- bake(prepped_recipe, new_data = dr_test)

# Feature matrix for SHAP (remove outcome)
X_train <- train_baked %>% select(-label) %>% as.data.frame()
X_test  <- test_baked  %>% select(-label) %>% as.data.frame()

# =============================================================================
# SECTION 1: Variable Importance — Random Forest & XGBoost
# =============================================================================
# Built-in importance: fast, model-specific, good for global overview

cat("── Variable Importance Plots ──\n")

# Prediction wrapper for vip (works with tidymodels workflows)
pred_wrapper <- function(object, newdata) {
  predict(object, newdata, type = "prob")[[".pred_DR Present"]]
}

# Random Forest — Gini impurity-based importance
vip_rf <- vip(
  best_rf,
  method      = "model",
  num_features = 15,
  aesthetics  = list(fill = "#74c476", color = "#238b45", alpha = 0.85)
) +
  labs(
    title    = "Random Forest — Variable Importance",
    subtitle = "Mean decrease in Gini impurity across trees",
    x        = "Feature",
    y        = "Importance (Gini)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# XGBoost — gain-based importance
xgb_engine  <- extract_fit_engine(best_xgb)
xgb_imp_raw <- xgb.importance(model = xgb_engine)

p_xgb_imp <- xgb_imp_raw %>%
  as_tibble() %>%
  slice_max(Gain, n = 15) %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(x = Feature, y = Gain)) +
  geom_col(fill = "#fd8d3c", color = "#d94801", alpha = 0.85) +
  coord_flip() +
  labs(
    title    = "XGBoost — Variable Importance",
    subtitle = "Feature gain: average improvement in splits using this feature",
    x        = NULL,
    y        = "Gain"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

p_vip_combined <- vip_rf / p_xgb_imp +
  plot_annotation(
    title   = "Feature Importance: RF vs XGBoost",
    caption = "Agreement between models on top features increases confidence"
  )

ggsave("outputs/figures/interp_variable_importance.png",
       p_vip_combined, width = 10, height = 12, dpi = 220)
cat("  Saved: interp_variable_importance.png\n")

# =============================================================================
# SECTION 2: SHAP Values — XGBoost (model-agnostic via kernelshap)
# =============================================================================
# SHAP (SHapley Additive exPlanations) decomposes each prediction into
# additive feature contributions — the gold standard for clinical AI explainability.
#
# For XGBoost we can use the faster treeshap implementation via shapviz.
# For other models, kernelshap provides a model-agnostic approximation.

cat("\n── SHAP Values (XGBoost — TreeSHAP) ──\n")

# TreeSHAP: exact SHAP for tree models — fast and exact
xgb_matrix_train <- xgboost::xgb.DMatrix(as.matrix(X_train))
xgb_matrix_test  <- xgboost::xgb.DMatrix(as.matrix(X_test))

# shapviz uses the raw xgboost model + data
shap_xgb <- shapviz(
  xgb_engine,
  X_pred = xgb_matrix_test,
  X      = X_test
)

# ── SHAP Plot 1: Summary plot (beeswarm) ─────────────────────────────────────
# Shows distribution of SHAP values per feature across all test patients.
# Color = feature value (red = high, blue = low).
# X position = SHAP value (right = pushes toward DR prediction).

p_shap_summary <- sv_importance(
  shap_xgb,
  kind        = "beeswarm",
  max_display = 14,
  alpha       = 0.6
) +
  labs(
    title    = "SHAP Summary — XGBoost",
    subtitle = paste(
      "Each point = one patient.",
      "X-axis: contribution to DR probability.",
      "Color: feature value (red=high, blue=low)."
    ),
    x = "SHAP Value (impact on model output)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/interp_shap_summary.png",
       p_shap_summary, width = 10, height = 8, dpi = 220)
cat("  Saved: interp_shap_summary.png\n")

# ── SHAP Plot 2: Bar plot (mean |SHAP|) ──────────────────────────────────────
p_shap_bar <- sv_importance(
  shap_xgb,
  kind        = "bar",
  max_display = 14,
  fill        = "#fd8d3c"
) +
  labs(
    title    = "Mean |SHAP| — Global Feature Importance",
    subtitle = "Average absolute SHAP value: higher = more influential overall",
    x        = "Mean |SHAP Value|"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/interp_shap_bar.png",
       p_shap_bar, width = 9, height = 7, dpi = 220)

# ── SHAP Plot 3: Dependence plots for top features ───────────────────────────
# Shows how SHAP value changes with feature value, colored by an interaction feature.
# Reveals non-linear effects and interactions — often surprising clinically.

top_features <- sv_importance(shap_xgb, kind = "bar", max_display = 4)$data %>%
  pull(feature) %>%
  as.character()

if (length(top_features) >= 4) {
  dep_plots <- map(top_features[1:4], function(feat) {
    sv_dependence(shap_xgb, v = feat, alpha = 0.5) +
      labs(title = feat) +
      theme_minimal(base_size = 11) +
      theme(plot.title = element_text(face = "bold", size = 11))
  })

  p_dependence <- wrap_plots(dep_plots, ncol = 2) +
    plot_annotation(
      title   = "SHAP Dependence Plots — Top 4 Features",
      subtitle = "How each feature's value maps to its contribution toward DR prediction",
      caption  = "Color indicates interaction with the most correlated other feature"
    )

  ggsave("outputs/figures/interp_shap_dependence.png",
         p_dependence, width = 12, height = 10, dpi = 220)
  cat("  Saved: interp_shap_dependence.png\n")
}

# =============================================================================
# SECTION 3: SHAP for Logistic Regression (kernelshap — model-agnostic)
# =============================================================================
# Demonstrates that SHAP interpretability works across ALL model types —
# important for clinical settings where a simpler, auditable model may be
# preferred over XGBoost.

cat("\n── SHAP Values (Logistic Regression — KernelSHAP) ──\n")
cat("  Computing KernelSHAP (using 100 background samples)...\n")

# Prediction function for LR workflow
lr_predict_fn <- function(model, newdata) {
  as.data.frame(newdata) %>%
    as_tibble() %>%
    predict(model, new_data = ., type = "prob") %>%
    pull(`.pred_DR Present`)
}

# Use a random subsample as background (KernelSHAP is expensive on full data)
set.seed(42)
background_idx <- sample(nrow(X_train), min(100, nrow(X_train)))
X_background   <- X_train[background_idx, ]

ks_lr <- kernelshap(
  best_lr,
  X          = X_test[1:min(100, nrow(X_test)), ],  # explain first 100 test obs
  bg_X       = X_background,
  pred_fun   = lr_predict_fn,
  verbose    = FALSE
)

shap_lr <- shapviz(ks_lr)

p_shap_lr <- sv_importance(shap_lr, kind = "beeswarm", max_display = 12,
                           alpha = 0.6) +
  labs(
    title    = "SHAP Summary — Logistic Regression",
    subtitle = "KernelSHAP (model-agnostic) on first 100 test patients",
    x        = "SHAP Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/interp_shap_lr.png",
       p_shap_lr, width = 10, height = 7, dpi = 220)
cat("  Saved: interp_shap_lr.png\n")

# =============================================================================
# SECTION 4: Individual Case Narratives
# =============================================================================
# Pick one True Positive and one False Negative from XGBoost predictions.
# Use SHAP waterfall plots to explain *why* the model got each one right/wrong.
# This directly addresses the clinical AI review question: "Why did it say that?"

cat("\n── Individual Case Narratives ──\n")

xgb_test_preds <- predict(best_xgb, dr_test, type = "prob") %>%
  bind_cols(predict(best_xgb, dr_test, type = "class")) %>%
  bind_cols(dr_test %>% select(label)) %>%
  rename(prob_dr = `.pred_DR Present`,
         predicted = .pred_class,
         truth = label) %>%
  mutate(row_idx = row_number())

# True Positive: correctly identified DR patient with high confidence
true_positive <- xgb_test_preds %>%
  filter(truth == "DR Present", predicted == "DR Present") %>%
  slice_max(prob_dr, n = 1)

# False Negative: DR patient the model missed (most confident wrong prediction)
false_negative <- xgb_test_preds %>%
  filter(truth == "DR Present", predicted == "No DR") %>%
  slice_min(prob_dr, n = 1)

cat(sprintf("  True Positive case:  row %d | predicted prob DR = %.3f\n",
            true_positive$row_idx, true_positive$prob_dr))

if (nrow(false_negative) > 0) {
  cat(sprintf("  False Negative case: row %d | predicted prob DR = %.3f\n",
              false_negative$row_idx, false_negative$prob_dr))
}

# Waterfall plot for True Positive
tp_shap <- shapviz(
  xgb_engine,
  X_pred = xgboost::xgb.DMatrix(
    as.matrix(X_test[true_positive$row_idx, , drop = FALSE])
  ),
  X = X_test[true_positive$row_idx, , drop = FALSE]
)

p_waterfall_tp <- sv_waterfall(tp_shap, row_id = 1) +
  labs(
    title    = "SHAP Waterfall — True Positive",
    subtitle = sprintf(
      "Patient correctly identified as DR (predicted prob = %.3f)",
      true_positive$prob_dr
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/figures/interp_waterfall_true_positive.png",
       p_waterfall_tp, width = 9, height = 7, dpi = 220)
cat("  Saved: interp_waterfall_true_positive.png\n")

# Waterfall for False Negative (if it exists)
if (nrow(false_negative) > 0) {
  fn_shap <- shapviz(
    xgb_engine,
    X_pred = xgboost::xgb.DMatrix(
      as.matrix(X_test[false_negative$row_idx, , drop = FALSE])
    ),
    X = X_test[false_negative$row_idx, , drop = FALSE]
  )

  p_waterfall_fn <- sv_waterfall(fn_shap, row_id = 1) +
    labs(
      title    = "SHAP Waterfall — False Negative (Missed DR)",
      subtitle = sprintf(
        "Patient with DR missed by model (predicted prob = %.3f) | Clinically dangerous",
        false_negative$prob_dr
      )
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(color = "#b2182b")
    )

  ggsave("outputs/figures/interp_waterfall_false_negative.png",
         p_waterfall_fn, width = 9, height = 7, dpi = 220)
  cat("  Saved: interp_waterfall_false_negative.png\n")
}

cat("\n── All interpretability figures saved to outputs/figures/ ──\n")
cat("Key files:\n")
cat("  interp_variable_importance.png   — RF vs XGBoost feature ranking\n")
cat("  interp_shap_summary.png          — SHAP beeswarm (XGBoost)\n")
cat("  interp_shap_dependence.png       — Feature-level SHAP dependence\n")
cat("  interp_shap_lr.png               — SHAP for logistic regression\n")
cat("  interp_waterfall_true_positive.png  — Individual TP explanation\n")
cat("  interp_waterfall_false_negative.png — Individual FN explanation\n")
