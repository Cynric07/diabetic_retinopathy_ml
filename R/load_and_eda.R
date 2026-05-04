# =============================================================================
# load_and_eda.R
# Load data, attach clinical feature glossary, run exploratory data analysis
#
# Inputs : data/raw/messidor_features.arff
# Outputs: outputs/figures/eda_*.png
#          dr_raw (data frame) — available to subsequent scripts
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(patchwork)      # composing multi-panel figures
  library(corrplot)       # correlation heatmap
  library(RWeka)          # reading .arff files
  library(scales)         # axis formatting
  library(ggridges)       # ridge plots
})

# ── 1. Load data ──────────────────────────────────────────────────────────────

arff_path <- "data/raw/messidor_features.arff"

if (!file.exists(arff_path)) {
  stop(
    "Dataset not found at '", arff_path, "'.\n",
    "Download from: https://archive.ics.uci.edu/dataset/329/",
    "diabetic+retinopathy+debrecen+data+set\n",
    "Save as: data/raw/messidor_features.arff"
  )
}

dr_raw <- RWeka::read.arff(arff_path)

# ── 2. Attach clinical feature names ─────────────────────────────────────────
# The UCI file uses generic V1..V20; we replace with meaningful clinical names

clinical_names <- c(
  "quality",          # 1  - binary quality assessment
  "pre_screening",    # 2  - binary pre-screening for severe abnormality
  "ma_detect_0.5",    # 3  - MA detections at confidence 0.5
  "ma_detect_0.6",    # 4
  "ma_detect_0.7",    # 5
  "ma_detect_0.8",    # 6
  "ma_detect_0.9",    # 7
  "ma_detect_1.0",    # 8
  "exudate_1",        # 9  - normalized exudate features
  "exudate_2",        # 10
  "exudate_3",        # 11
  "exudate_4",        # 12
  "exudate_5",        # 13
  "exudate_6",        # 14
  "exudate_7",        # 15
  "exudate_8",        # 16
  "macula_od_dist",   # 17 - macula-to-optic-disc distance (normalized)
  "optic_disc_diam",  # 18 - optic disc diameter
  "amfm_class",       # 19 - AM/FM classification result
  "label"             # 20 - TARGET: 1 = DR present, 0 = no DR
)

colnames(dr_raw) <- clinical_names

# Convert label to factor with descriptive levels
dr_raw <- dr_raw %>%
  mutate(
    label = factor(label, levels = c(0, 1),
                   labels = c("No DR", "DR Present")),
    quality       = factor(quality),
    pre_screening = factor(pre_screening),
    amfm_class    = factor(amfm_class)
  )

cat("Dataset loaded:", nrow(dr_raw), "patients,", ncol(dr_raw), "features\n")

# ── 3. Basic data quality checks ─────────────────────────────────────────────

cat("\n── Data Quality Summary ──\n")
cat("Missing values per column:\n")
missing_summary <- colSums(is.na(dr_raw))
print(missing_summary[missing_summary > 0])
if (sum(missing_summary) == 0) cat("  → No missing values detected ✓\n")

cat("\nClass distribution:\n")
print(table(dr_raw$label))
cat(sprintf("  Positive rate: %.1f%%\n",
            mean(dr_raw$label == "DR Present") * 100))

# ── 4. EDA Plot 1: Class distribution ────────────────────────────────────────

p_class <- dr_raw %>%
  count(label) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = label, y = n, fill = label)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
            vjust = -0.3, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("No DR" = "#4393c3", "DR Present" = "#d6604d")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Class Distribution",
    subtitle = "Moderate imbalance — ~54% positive rate",
    x        = NULL,
    y        = "Patient Count"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

# ── 5. EDA Plot 2: MA detection features by class ────────────────────────────

ma_features <- dr_raw %>%
  select(label, starts_with("ma_detect")) %>%
  pivot_longer(-label, names_to = "confidence", values_to = "detections") %>%
  mutate(confidence = str_replace(confidence, "ma_detect_", "α = "))

p_ma <- ggplot(ma_features, aes(x = detections, y = confidence,
                                 fill = label, color = label)) +
  geom_density_ridges(alpha = 0.6, scale = 1.2, rel_min_height = 0.01) +
  scale_fill_manual(values  = c("No DR" = "#4393c3", "DR Present" = "#d6604d")) +
  scale_color_manual(values = c("No DR" = "#2166ac", "DR Present" = "#b2182b")) +
  scale_x_continuous(limits = c(0, NA)) +
  labs(
    title    = "Microaneurysm Detections by Confidence Level",
    subtitle = "Higher confidence thresholds → fewer but more specific detections",
    x        = "Number of MAs Detected",
    y        = "Detection Confidence Threshold",
    fill     = NULL, color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title   = element_text(face = "bold"),
    legend.position = "bottom"
  )

# ── 6. EDA Plot 3: Exudate features boxplots ─────────────────────────────────

exudate_features <- dr_raw %>%
  select(label, starts_with("exudate")) %>%
  pivot_longer(-label, names_to = "feature", values_to = "value") %>%
  mutate(feature = str_replace(feature, "exudate_", "Exudate "))

p_exudate <- ggplot(exudate_features,
                    aes(x = feature, y = value, fill = label)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.4, width = 0.65) +
  scale_fill_manual(values = c("No DR" = "#4393c3", "DR Present" = "#d6604d")) +
  scale_y_log10(labels = label_number()) +
  labs(
    title    = "Exudate Feature Distributions by Class",
    subtitle = "Log scale — exudates signal vascular leakage characteristic of DR",
    x        = NULL,
    y        = "Feature Value (log scale)",
    fill     = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold"),
    axis.text.x     = element_text(angle = 35, hjust = 1),
    legend.position = "bottom"
  )

# ── 7. EDA Plot 4: Macula distance & optic disc by class ────────────────────

p_macula <- dr_raw %>%
  select(label, macula_od_dist, optic_disc_diam) %>%
  pivot_longer(-label, names_to = "feature", values_to = "value") %>%
  mutate(feature = recode(feature,
    "macula_od_dist"  = "Macula–OD Distance (normalized)",
    "optic_disc_diam" = "Optic Disc Diameter"
  )) %>%
  ggplot(aes(x = value, fill = label)) +
  geom_histogram(bins = 40, alpha = 0.7, position = "identity") +
  facet_wrap(~feature, scales = "free") +
  scale_fill_manual(values = c("No DR" = "#4393c3", "DR Present" = "#d6604d")) +
  labs(
    title    = "Structural Retinal Features",
    subtitle = "Macula-optic disc geometry and disc size",
    x = "Value", y = "Count", fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold"),
    legend.position = "bottom"
  )

# ── 8. EDA Plot 5: Correlation heatmap (numeric features only) ───────────────

numeric_features <- dr_raw %>%
  select(where(is.numeric)) %>%
  # exclude label-encoded variables; keep continuous
  select(starts_with("ma_detect"), starts_with("exudate"),
         macula_od_dist, optic_disc_diam)

cor_matrix <- cor(numeric_features, use = "complete.obs", method = "spearman")

png("outputs/figures/eda_correlation_heatmap.png",
    width = 2400, height = 2200, res = 220)
corrplot(
  cor_matrix,
  method   = "color",
  type     = "upper",
  tl.cex   = 0.75,
  tl.col   = "black",
  addCoef.col = "black",
  number.cex  = 0.55,
  col      = colorRampPalette(c("#2166ac", "white", "#b2182b"))(200),
  title    = "Spearman Correlation — Retinal Features",
  mar      = c(0, 0, 2, 0)
)
dev.off()

# ── 9. Save composite EDA figure ─────────────────────────────────────────────

combined_eda <- (p_class | p_macula) / p_ma / p_exudate +
  plot_annotation(
    title   = "Diabetic Retinopathy Dataset — Exploratory Data Analysis",
    caption = "Data: Antal & Hajdu (2014), UCI ML Repository",
    theme   = theme(plot.title = element_text(size = 15, face = "bold"))
  )

ggsave("outputs/figures/eda_overview.png",
       combined_eda, width = 16, height = 18, dpi = 220)

cat("\n── EDA figures saved to outputs/figures/ ──\n")

# ── 10. Save cleaned data object for downstream scripts ──────────────────────
# (subsequent scripts load this via source("R/load_and_eda.R"))
# dr_raw stays in the global environment

cat("  dr_raw object available for downstream scripts\n")
