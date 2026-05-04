# =============================================================================
# evaluation.R
# Clinical evaluation: ROC, PR curves, confusion matrices, calibration,
# threshold analysis, and sensitivity/specificity narrative
#
# Requires: best_lr, best_rf, best_xgb, best_svm, dr_test (from prior scripts)
# Outputs : outputs/figures/eval_*.png
#           test_results (tibble with predictions + metrics)
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(patchwork)
  library(probably)     # calibration plots
  library(scales)
  library(gt)           # publication-quality tables
})

# ── 1. Generate test-set predictions for all models ──────────────────────────

get_test_preds <- function(fitted_wf, model_name, test_data) {
  pred_class <- predict(fitted_wf, test_data, type = "class")
  pred_prob  <- predict(fitted_wf, test_data, type = "prob")

  bind_cols(
    test_data %>% select(label),
    pred_class,
    pred_prob
  ) %>%
    mutate(model = model_name)
}

all_preds <- bind_rows(
  get_test_preds(best_lr,  "Logistic Regression", dr_test),
  get_test_preds(best_rf,  "Random Forest",        dr_test),
  get_test_preds(best_xgb, "XGBoost",              dr_test),
  get_test_preds(best_svm, "SVM",                  dr_test)
)

# Rename for clarity
all_preds <- all_preds %>%
  rename(
    truth        = label,
    predicted    = .pred_class,
    prob_no_dr   = `.pred_No DR`,
    prob_dr      = `.pred_DR Present`
  ) %>%
  mutate(model = factor(model,
    levels = c("Logistic Regression", "Random Forest", "XGBoost", "SVM")))

# ── 2. Summary metrics at default threshold (0.5) ────────────────────────────

model_colors <- c(
  "Logistic Regression" = "#4393c3",
  "Random Forest"       = "#74c476",
  "XGBoost"             = "#fd8d3c",
  "SVM"                 = "#9e9ac8"
)

compute_metrics <- function(preds) {
  preds %>%
    group_by(model) %>%
    summarise(
      AUROC       = roc_auc_vec(truth, prob_dr, event_level = "second"),
      Sensitivity = sens_vec(truth, predicted, event_level = "second"),
      Specificity = spec_vec(truth, predicted, event_level = "second"),
      F1          = f_meas_vec(truth, predicted, event_level = "second"),
      Accuracy    = accuracy_vec(truth, predicted),
      .groups     = "drop"
    )
}

metrics_default <- compute_metrics(all_preds)
cat("── Test Set Metrics (threshold = 0.5) ──\n")
print(metrics_default %>% mutate(across(where(is.numeric), ~round(.x, 4))))

# ── 3. ROC Curves ─────────────────────────────────────────────────────────────

roc_data <- all_preds %>%
  group_by(model) %>%
  roc_curve(truth, prob_dr, event_level = "second") %>%
  ungroup()

# Compute AUC per model for legend labels
auc_labels <- metrics_default %>%
  mutate(label = sprintf("%s (AUC = %.3f)", model, AUROC)) %>%
  select(model, label)

roc_data_labeled <- roc_data %>%
  left_join(auc_labels, by = "model")

p_roc <- ggplot(roc_data_labeled,
                aes(x = 1 - specificity, y = sensitivity,
                    color = label, group = model)) +
  geom_line(linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey60", linewidth = 0.7) +
  # Mark 90% sensitivity threshold points
  geom_hline(yintercept = 0.90, linetype = "dotted",
             color = "#b2182b", linewidth = 0.8) +
  annotate("text", x = 0.85, y = 0.91, label = "90% sensitivity",
           color = "#b2182b", size = 3.5, hjust = 1) +
  scale_color_manual(values = c(
    grep("Logistic", unique(roc_data_labeled$label), value = TRUE) ~ "#4393c3",
    setNames(
      c("#4393c3", "#74c476", "#fd8d3c", "#9e9ac8"),
      unique(roc_data_labeled$label)
    )
  )) +
  scale_color_brewer(palette = "Set1") +
  coord_equal() +
  labs(
    title    = "ROC Curves — Test Set",
    subtitle = "Red dotted line: clinically motivated 90% sensitivity threshold",
    x        = "1 − Specificity (False Positive Rate)",
    y        = "Sensitivity (True Positive Rate)",
    color    = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold"),
    legend.position = c(0.65, 0.25),
    legend.background = element_rect(fill = "white", color = "grey80")
  )

# ── 4. Precision-Recall Curves ────────────────────────────────────────────────
# More informative than ROC under class imbalance — shows the tradeoff between
# precision (PPV) and recall (sensitivity) at every threshold.

pr_data <- all_preds %>%
  group_by(model) %>%
  pr_curve(truth, prob_dr, event_level = "second") %>%
  ungroup()

p_pr <- ggplot(pr_data, aes(x = recall, y = precision, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_hline(
    yintercept = mean(dr_test$label == "DR Present"),
    linetype   = "dashed",
    color      = "grey50",
    linewidth  = 0.7
  ) +
  annotate("text", x = 0.95, y = mean(dr_test$label == "DR Present") + 0.02,
           label = "Baseline (prevalence)", size = 3.2, hjust = 1,
           color = "grey40") +
  scale_color_brewer(palette = "Set1") +
  labs(
    title    = "Precision-Recall Curves — Test Set",
    subtitle = "Dashed: baseline precision if predicting all positive",
    x        = "Recall (Sensitivity)",
    y        = "Precision (PPV)",
    color    = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold"),
    legend.position = "bottom"
  )

# ── 5. Confusion matrices at 90% sensitivity threshold ───────────────────────
# Find the threshold for each model that achieves ≥ 90% sensitivity

find_sensitivity_threshold <- function(preds, target_sens = 0.90) {
  thresholds <- seq(0.01, 0.99, by = 0.01)
  results <- map_dfr(thresholds, function(t) {
    pred_at_t <- factor(
      ifelse(preds$prob_dr >= t, "DR Present", "No DR"),
      levels = c("No DR", "DR Present")
    )
    tibble(
      threshold   = t,
      sensitivity = sens_vec(preds$truth, pred_at_t, event_level = "second"),
      specificity = spec_vec(preds$truth, pred_at_t, event_level = "second")
    )
  })
  # Find the highest threshold that still meets sensitivity target
  results %>%
    filter(sensitivity >= target_sens) %>%
    slice_max(threshold, n = 1)
}

thresholds_90 <- all_preds %>%
  group_by(model) %>%
  group_modify(~find_sensitivity_threshold(.x)) %>%
  ungroup()

cat("\n── Sensitivity-Optimized Thresholds (≥90% sensitivity) ──\n")
print(thresholds_90 %>% mutate(across(where(is.numeric), ~round(.x, 4))))

# Confusion matrices at optimized thresholds
conf_mat_plots <- map(unique(as.character(all_preds$model)), function(m) {
  thresh <- thresholds_90 %>% filter(model == m) %>% pull(threshold)
  preds_m <- all_preds %>% filter(model == m)

  pred_optimized <- factor(
    ifelse(preds_m$prob_dr >= thresh, "DR Present", "No DR"),
    levels = c("No DR", "DR Present")
  )

  cm <- conf_mat(
    data.frame(truth = preds_m$truth, estimate = pred_optimized),
    truth   = truth,
    estimate = estimate
  )

  autoplot(cm, type = "heatmap") +
    scale_fill_gradient(low = "#deebf7", high = "#08519c") +
    labs(
      title    = m,
      subtitle = sprintf("Threshold = %.2f (sens ≥ 90%%)", thresh)
    ) +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11),
          legend.position = "none")
})

p_conf_mats <- wrap_plots(conf_mat_plots, ncol = 2) +
  plot_annotation(
    title   = "Confusion Matrices at Sensitivity-Optimized Thresholds",
    caption = "Optimized for ≥90% sensitivity (minimizing false negatives / missed DR cases)"
  )

# ── 6. Threshold sensitivity/specificity tradeoff plot ───────────────────────

threshold_sweep <- all_preds %>%
  group_by(model) %>%
  group_modify(function(preds, key) {
    map_dfr(seq(0.05, 0.95, by = 0.02), function(t) {
      pred_t <- factor(
        ifelse(preds$prob_dr >= t, "DR Present", "No DR"),
        levels = c("No DR", "DR Present")
      )
      tibble(
        threshold   = t,
        sensitivity = sens_vec(preds$truth, pred_t, event_level = "second"),
        specificity = spec_vec(preds$truth, pred_t, event_level = "second")
      )
    })
  }) %>%
  ungroup() %>%
  pivot_longer(c(sensitivity, specificity),
               names_to = "metric", values_to = "value")

p_threshold <- ggplot(threshold_sweep,
                      aes(x = threshold, y = value,
                          color = metric, linetype = model)) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_vline(xintercept = 0.5, linetype = "dotted",
             color = "grey50", linewidth = 0.7) +
  scale_color_manual(values = c(sensitivity = "#d6604d",
                                specificity = "#4393c3")) +
  facet_wrap(~model, ncol = 2) +
  labs(
    title    = "Sensitivity–Specificity Tradeoff by Threshold",
    subtitle = "In screening: prioritize sensitivity — missing DR risks preventable blindness",
    x        = "Classification Threshold",
    y        = "Metric Value",
    color    = NULL,
    linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold"),
    legend.position = "bottom"
  )

# ── 7. Calibration plot (XGBoost — typically best calibrated) ────────────────
# Calibration checks: does "80% predicted probability" actually mean
# ~80% of those patients have DR? Essential for clinical use.

xgb_preds_cal <- all_preds %>%
  filter(model == "XGBoost") %>%
  select(truth, prob_dr)

p_calibration <- xgb_preds_cal %>%
  mutate(bin = cut(prob_dr, breaks = seq(0, 1, by = 0.1),
                   include.lowest = TRUE, right = TRUE)) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(prob_dr),
    obs_rate  = mean(truth == "DR Present"),
    n         = n(),
    .groups   = "drop"
  ) %>%
  ggplot(aes(x = mean_pred, y = obs_rate)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey60", linewidth = 0.8) +
  geom_point(aes(size = n), color = "#fd8d3c", alpha = 0.85) +
  geom_line(color = "#fd8d3c", linewidth = 0.9) +
  scale_size_continuous(range = c(2, 8), name = "n patients") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title    = "Calibration Plot — XGBoost",
    subtitle = "Points on the diagonal = perfectly calibrated probabilities",
    x        = "Mean Predicted Probability",
    y        = "Observed DR Rate"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

# ── 8. Save all evaluation figures ───────────────────────────────────────────

ggsave("outputs/figures/eval_roc_curves.png",       p_roc,         width = 8,  height = 7,  dpi = 220)
ggsave("outputs/figures/eval_pr_curves.png",        p_pr,          width = 8,  height = 6,  dpi = 220)
ggsave("outputs/figures/eval_confusion_matrices.png", p_conf_mats, width = 10, height = 9,  dpi = 220)
ggsave("outputs/figures/eval_threshold_tradeoff.png", p_threshold, width = 12, height = 8,  dpi = 220)
ggsave("outputs/figures/eval_calibration.png",      p_calibration, width = 7,  height = 7,  dpi = 220)

# ── 9. Final results table ────────────────────────────────────────────────────

# Merge default and optimized-threshold metrics
metrics_optimized <- map_dfr(unique(as.character(all_preds$model)), function(m) {
  thresh <- thresholds_90 %>% filter(model == m) %>% pull(threshold)
  preds_m <- all_preds %>% filter(model == m)

  pred_opt <- factor(
    ifelse(preds_m$prob_dr >= thresh, "DR Present", "No DR"),
    levels = c("No DR", "DR Present")
  )
  tibble(
    model       = m,
    threshold   = thresh,
    sensitivity = sens_vec(preds_m$truth, pred_opt, event_level = "second"),
    specificity = spec_vec(preds_m$truth, pred_opt, event_level = "second"),
    f1          = f_meas_vec(preds_m$truth, pred_opt, event_level = "second")
  )
})

test_results <- metrics_default %>%
  left_join(metrics_optimized,
            by = "model",
            suffix = c("_default", "_opt")) %>%
  mutate(across(where(is.numeric), ~round(.x, 4)))

saveRDS(test_results, "outputs/models/test_results.rds")

cat("\n── Final Test Results Summary ──\n")
print(test_results)
cat("\nEvaluation figures saved to outputs/figures/\n")
