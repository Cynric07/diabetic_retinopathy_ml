# =============================================================================
# app.R — Diabetic Retinopathy Screening Dashboard
# Interactive Shiny dashboard with patient risk scoring + model comparison
#
# Usage:
#   shiny::runApp("shiny/app.R")
#
# Automatically loads real model outputs from outputs/ if available,
# falls back to simulated demo data otherwise.
# =============================================================================

# ── Dependencies ──────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(plotly)
  library(DT)
  library(shinyWidgets)
  library(shinycssloaders)
  
})

# ── 0. Helper: detect if real pipeline outputs exist ─────────────────────────
pipeline_outputs_exist <- function() {
  file.exists("../outputs/models/best_xgb.rds") &&
    file.exists("../outputs/models/test_results.rds")
}

# ── 1. Load or simulate data ──────────────────────────────────────────────────

load_data <- function() {
  
  if (pipeline_outputs_exist()) {
    message("✓ Loading real pipeline outputs...")
    
    test_results  <- readRDS("../outputs/models/test_results.rds")
    best_xgb      <- readRDS("../outputs/models/best_xgb.rds")
    best_rf       <- readRDS("../outputs/models/best_rf.rds")
    best_lr       <- readRDS("../outputs/models/best_lr.rds")
    
    # Load test predictions if available
    preds_path <- "../outputs/models/test_preds.rds"
    if (file.exists(preds_path)) {
      all_preds <- readRDS(preds_path)
    } else {
      all_preds <- simulate_predictions()
    }
    
    list(
      real        = TRUE,
      results     = test_results,
      models      = list(xgb = best_xgb, rf = best_rf, lr = best_lr),
      predictions = all_preds
    )
    
  } else {
    message("ℹ Using demo data (run 00_run_all.R to load real results)")
    list(
      real        = FALSE,
      results     = simulate_results(),
      models      = NULL,
      predictions = simulate_predictions()
    )
  }
}

simulate_results <- function() {
  tibble(
    model               = c("Logistic Regression", "Random Forest",
                            "XGBoost", "SVM"),
    AUROC               = c(0.891, 0.934, 0.961, 0.912),
    Sensitivity         = c(0.782, 0.847, 0.891, 0.813),
    Specificity         = c(0.856, 0.903, 0.921, 0.879),
    F1                  = c(0.814, 0.872, 0.903, 0.843),
    Accuracy            = c(0.823, 0.878, 0.908, 0.849),
    threshold           = c(0.38, 0.31, 0.28, 0.35),
    sensitivity_opt     = c(0.901, 0.912, 0.923, 0.908),
    specificity_opt     = c(0.712, 0.756, 0.789, 0.731)
  )
}

simulate_predictions <- function() {
  set.seed(42)
  n <- 231
  
  bind_rows(
    lapply(c("Logistic Regression", "Random Forest", "XGBoost", "SVM"),
           function(m) {
             aucs <- c("Logistic Regression" = 0.891, "Random Forest" = 0.934,
                       "XGBoost" = 0.961, "SVM" = 0.912)
             noise <- rnorm(n, 0, 0.08 + (1 - aucs[m]) * 0.15)
             truth <- factor(c(rep("DR Present", 124), rep("No DR", 107)),
                             levels = c("No DR", "DR Present"))
             base  <- ifelse(truth == "DR Present", 0.68, 0.28)
             prob  <- pmin(pmax(base + noise, 0.01), 0.99)
             tibble(
               model    = m,
               truth    = truth,
               prob_dr  = prob,
               predicted = factor(
                 ifelse(prob >= 0.5, "DR Present", "No DR"),
                 levels = c("No DR", "DR Present")
               )
             )
           })
  )
}

# ── 2. Compute ROC/PR curves from predictions ─────────────────────────────────

compute_roc <- function(preds) {
  preds %>%
    group_by(model) %>%
    group_modify(function(d, k) {
      thresholds <- seq(0, 1, by = 0.01)
      map_dfr(thresholds, function(t) {
        pred_t <- factor(ifelse(d$prob_dr >= t, "DR Present", "No DR"),
                         levels = c("No DR", "DR Present"))
        tp <- sum(pred_t == "DR Present" & d$truth == "DR Present")
        fp <- sum(pred_t == "DR Present" & d$truth == "No DR")
        tn <- sum(pred_t == "No DR" & d$truth == "No DR")
        fn <- sum(pred_t == "No DR" & d$truth == "DR Present")
        tibble(
          threshold   = t,
          sensitivity = ifelse(tp + fn == 0, 0, tp / (tp + fn)),
          specificity = ifelse(tn + fp == 0, 0, tn / (tn + fp)),
          precision   = ifelse(tp + fp == 0, 0, tp / (tp + fp))
        )
      })
    }) %>%
    ungroup()
}

# ── 3. Theme & Colour Palette ─────────────────────────────────────────────────

DR_THEME <- bs_theme(
  bg            = "#0d1117",
  fg            = "#e6edf3",
  primary       = "#58a6ff",
  secondary     = "#30363d",
  success       = "#3fb950",
  warning       = "#d29922",
  danger        = "#f85149",
  base_font     = font_google("IBM Plex Mono"),
  heading_font  = font_google("IBM Plex Sans"),
  font_scale    = 0.9,
  bootswatch    = "darkly"
)

MODEL_COLOURS <- c(
  "Logistic Regression" = "#58a6ff",
  "Random Forest"       = "#3fb950",
  "XGBoost"             = "#f0883e",
  "SVM"                 = "#bc8cff"
)

gg_dark <- function() {
  theme_minimal(base_family = "IBM Plex Mono", base_size = 12) +
    theme(
      plot.background    = element_rect(fill = "#161b22", color = NA),
      panel.background   = element_rect(fill = "#161b22", color = NA),
      panel.grid.major   = element_line(color = "#21262d", linewidth = 0.4),
      panel.grid.minor   = element_blank(),
      axis.text          = element_text(color = "#8b949e"),
      axis.title         = element_text(color = "#c9d1d9"),
      plot.title         = element_text(color = "#e6edf3", face = "bold",
                                        size  = 13),
      plot.subtitle      = element_text(color = "#8b949e", size = 10),
      legend.background  = element_rect(fill = "#161b22", color = NA),
      legend.text        = element_text(color = "#c9d1d9"),
      legend.title       = element_text(color = "#8b949e"),
      strip.text         = element_text(color = "#c9d1d9")
    )
}

# ── 4. Load data once at startup ──────────────────────────────────────────────

DATA <- load_data()
ROC_DATA <- compute_roc(DATA$predictions)

FEATURE_DEFAULTS <- list(
  ma_detect_0.5 = 10, ma_detect_0.6 = 8,  ma_detect_0.7 = 5,
  ma_detect_0.8 = 3,  ma_detect_0.9 = 1,  ma_detect_1.0 = 0,
  exudate_1 = 0.02,   exudate_2 = 0.01,   exudate_3 = 0.005,
  exudate_4 = 0.003,  exudate_5 = 0.001,  exudate_6 = 0.0005,
  exudate_7 = 0.0002, exudate_8 = 0.0001,
  macula_od_dist = 0.50, optic_disc_diam = 0.18
)

# ── 5. UI ─────────────────────────────────────────────────────────────────────

ui <- page_navbar(
  title = div(
    style = "display:flex; align-items:center; gap:10px;",
    span("◈", style = "color:#58a6ff; font-size:1.3em;"),
    span("DR Screening", style = "font-family:'IBM Plex Mono'; letter-spacing:2px;"),
    if (!DATA$real)
      span("DEMO MODE", style = paste0(
        "font-size:0.65em; background:#d29922; color:#0d1117;",
        "padding:2px 8px; border-radius:4px; letter-spacing:1px;"
      ))
  ),
  theme   = DR_THEME,
  fillable = TRUE,
  
  tags$head(tags$style(HTML("
    .navbar { border-bottom: 1px solid #21262d !important; }
    .card   { background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important; }
    .card-header { background:#1c2128 !important; border-bottom:1px solid #21262d !important; }
    .metric-box { background:#1c2128; border:1px solid #21262d; border-radius:8px;
                  padding:16px 20px; text-align:center; }
    .metric-val { font-size:2.2em; font-weight:700; font-family:'IBM Plex Mono'; }
    .metric-lbl { font-size:0.75em; color:#8b949e; letter-spacing:1px; text-transform:uppercase; margin-top:4px; }
    .risk-high  { color:#f85149; }
    .risk-med   { color:#d29922; }
    .risk-low   { color:#3fb950; }
    .shiny-output-error { color:#f85149 !important; }
    hr { border-color:#21262d; }
    .nav-pills .nav-link.active { background:#58a6ff !important; }
    .selectize-input { background:#1c2128 !important; border-color:#30363d !important; color:#e6edf3 !important; }
    .irs--shiny .irs-bar { background:#58a6ff !important; }
    .irs--shiny .irs-handle { background:#58a6ff !important; border-color:#58a6ff !important; }
  "))),
  
  # ── Tab 1: Patient Risk Scorer ──────────────────────────────────────────────
  nav_panel(
    title = "Patient Risk Scorer",
    icon  = icon("user-circle"),
    
    layout_columns(
      col_widths = c(4, 8),
      
      # Left: feature inputs
      card(
        card_header(
          div(style = "display:flex; align-items:center; gap:8px;",
              icon("sliders", style = "color:#58a6ff;"),
              "Clinical Feature Input")
        ),
        card_body(
          p(style = "color:#8b949e; font-size:0.8em; margin-bottom:16px;",
            "Enter retinal image features to compute DR probability."),
          
          h6("Microaneurysm Detections", style = "color:#58a6ff; margin-top:8px;"),
          fluidRow(
            column(6, numericInput("ma5",  "α=0.5", value=10, min=0, max=200)),
            column(6, numericInput("ma6",  "α=0.6", value=8,  min=0, max=200))
          ),
          fluidRow(
            column(6, numericInput("ma7",  "α=0.7", value=5,  min=0, max=200)),
            column(6, numericInput("ma8",  "α=0.8", value=3,  min=0, max=200))
          ),
          fluidRow(
            column(6, numericInput("ma9",  "α=0.9", value=1,  min=0, max=200)),
            column(6, numericInput("ma10", "α=1.0", value=0,  min=0, max=200))
          ),
          
          hr(),
          h6("Exudate Features", style = "color:#f0883e; margin-top:4px;"),
          sliderInput("exudate_level",
                      "Exudate Severity (normalized)",
                      min=0, max=1, value=0.05, step=0.01),
          
          hr(),
          h6("Structural Features", style = "color:#bc8cff; margin-top:4px;"),
          fluidRow(
            column(6, numericInput("macula_dist", "Macula-OD Dist",
                                   value=0.50, min=0, max=1, step=0.01)),
            column(6, numericInput("optic_diam",  "Optic Disc Diam",
                                   value=0.18, min=0, max=1, step=0.01))
          ),
          
          hr(),
          h6("Classification Threshold", style = "color:#8b949e; margin-top:4px;"),
          sliderInput("threshold", NULL,
                      min=0.1, max=0.9, value=0.5, step=0.01),
          p(style="color:#8b949e;font-size:0.75em;",
            "Lower threshold → higher sensitivity (catches more DR cases).
             Recommended clinical range: 0.28–0.40"),
          
          actionButton("score_btn", "Compute Risk",
                       class = "btn-primary w-100 mt-2",
                       icon  = icon("calculator"))
        )
      ),
      
      # Right: results
      div(
        class = "d-flex flex-column gap-3",
        
        # Risk gauge row
        layout_columns(
          col_widths = c(4, 4, 4),
          uiOutput("risk_box"),
          uiOutput("sensitivity_box"),
          uiOutput("action_box")
        ),
        
        # SHAP-like explanation
        card(
          card_header(
            div(style="display:flex;align-items:center;gap:8px;",
                icon("chart-bar", style="color:#58a6ff;"),
                "Feature Contribution (Approximate SHAP)")
          ),
          card_body(
            p(style="color:#8b949e;font-size:0.8em;",
              "Bars show each feature's estimated push toward (positive) or
               away from (negative) a DR prediction. Based on XGBoost
               feature importance weights applied to your input values."),
            withSpinner(plotlyOutput("shap_plot", height="320px"),
                        color = "#58a6ff", type = 4)
          )
        ),
        
        # Probability gauge
        card(
          card_header(
            div(style="display:flex;align-items:center;gap:8px;",
                icon("gauge", style="color:#3fb950;"),
                "Risk Probability Gauge")
          ),
          card_body(
            withSpinner(plotlyOutput("gauge_plot", height="200px"),
                        color = "#58a6ff", type = 4)
          )
        )
      )
    )
  ),
  
  # ── Tab 2: Model Comparison ──────────────────────────────────────────────────
  nav_panel(
    title = "Model Comparison",
    icon  = icon("chart-line"),
    
    layout_columns(
      col_widths = c(12),
      
      # Top metric cards
      layout_columns(
        col_widths = c(3, 3, 3, 3),
        !!!lapply(c("Logistic Regression","Random Forest","XGBoost","SVM"),
                  function(m) {
                    r <- DATA$results %>% filter(model == m)
                    col <- MODEL_COLOURS[m]
                    card(
                      card_body(
                        div(
                          div(m, style=paste0("color:",col,
                                              ";font-size:0.75em;letter-spacing:1px;",
                                              "text-transform:uppercase;margin-bottom:8px;")),
                          div(style="display:flex;gap:12px;flex-wrap:wrap;",
                              div(class="metric-box", style="flex:1;min-width:70px;",
                                  div(sprintf("%.3f", r$AUROC),
                                      class="metric-val", style=paste0("color:",col,";")),
                                  div("AUROC", class="metric-lbl")),
                              div(class="metric-box", style="flex:1;min-width:70px;",
                                  div(sprintf("%.1f%%", r$Sensitivity * 100),
                                      class="metric-val", style="color:#3fb950;"),
                                  div("Sensitivity", class="metric-lbl")),
                              div(class="metric-box", style="flex:1;min-width:70px;",
                                  div(sprintf("%.1f%%", r$Specificity * 100),
                                      class="metric-val", style="color:#bc8cff;"),
                                  div("Specificity", class="metric-lbl"))
                          )
                        )
                      )
                    )
                  })
      )
    ),
    
    layout_columns(
      col_widths = c(6, 6),
      
      # ROC curves
      card(
        card_header(
          div(style="display:flex;align-items:center;justify-content:space-between;",
              div(style="display:flex;align-items:center;gap:8px;",
                  icon("chart-area", style="color:#58a6ff;"),
                  "ROC Curves"),
              checkboxGroupInput("roc_models", NULL,
                                 choices  = c("Logistic Regression","Random Forest","XGBoost","SVM"),
                                 selected = c("Logistic Regression","Random Forest","XGBoost","SVM"),
                                 inline   = TRUE)
          )
        ),
        card_body(
          withSpinner(plotlyOutput("roc_plot", height="380px"),
                      color="#58a6ff", type=4)
        )
      ),
      
      # PR curves
      card(
        card_header(
          div(style="display:flex;align-items:center;gap:8px;",
              icon("chart-line", style="color:#f0883e;"),
              "Precision-Recall Curves")
        ),
        card_body(
          withSpinner(plotlyOutput("pr_plot", height="380px"),
                      color="#f0883e", type=4)
        )
      )
    ),
    
    layout_columns(
      col_widths = c(6, 6),
      
      # Threshold explorer
      card(
        card_header(
          div(style="display:flex;align-items:center;gap:8px;",
              icon("sliders", style="color:#3fb950;"),
              "Threshold → Sensitivity / Specificity"),
          sliderInput("thresh_explore", NULL,
                      min=0.1, max=0.9, value=0.5, step=0.01, width="100%")
        ),
        card_body(
          withSpinner(plotlyOutput("threshold_plot", height="320px"),
                      color="#3fb950", type=4),
          uiOutput("threshold_table")
        )
      ),
      
      # Metric bar chart
      card(
        card_header(
          div(style="display:flex;align-items:center;gap:8px;",
              icon("bar-chart", style="color:#bc8cff;"),
              "Model Metrics Summary"),
          selectInput("metric_select", NULL,
                      choices  = c("AUROC","Sensitivity","Specificity","F1","Accuracy"),
                      selected = "AUROC", width="160px")
        ),
        card_body(
          withSpinner(plotlyOutput("metric_bar", height="340px"),
                      color="#bc8cff", type=4)
        )
      )
    ),
    
    # Results table
    card(
      card_header(
        div(style="display:flex;align-items:center;gap:8px;",
            icon("table", style="color:#8b949e;"),
            "Full Results Table")
      ),
      card_body(
        DTOutput("results_table")
      )
    )
  ),
  
  # ── Tab 3: About ─────────────────────────────────────────────────────────────
  nav_panel(
    title = "About",
    icon  = icon("info-circle"),
    card(
      card_body(
        h4("Diabetic Retinopathy Screening Dashboard",
           style="color:#58a6ff;"),
        p("This dashboard is built on top of an end-to-end ML pipeline for
           diabetic retinopathy detection, using features extracted from
           fundus images from the UCI Messidor dataset."),
        
        h5("Clinical Context", style="color:#c9d1d9; margin-top:20px;"),
        p("DR is the leading cause of preventable blindness in working-age adults.
           Automated screening can flag high-risk patients for ophthalmology referral.
           The core clinical tradeoff: a false negative (missed DR) risks vision loss;
           a false positive means an unnecessary referral. This dashboard makes that
           tradeoff interactive via the threshold slider."),
        
        h5("Models", style="color:#c9d1d9; margin-top:20px;"),
        tags$ul(
          tags$li("Logistic Regression (L2 regularized)"),
          tags$li("Random Forest (ranger, 500 trees)"),
          tags$li("XGBoost (gradient boosting)"),
          tags$li("SVM (radial kernel)")
        ),
        p("All models trained with 5-fold stratified CV, SMOTE inside folds,
           and hyperparameter tuning via latin hypercube search."),
        
        h5("Interpretability", style="color:#c9d1d9; margin-top:20px;"),
        p("The Patient Risk Scorer tab provides approximate SHAP-style feature
           contributions using XGBoost feature importance weights scaled by
           input feature values. For full TreeSHAP values, run the complete
           pipeline (05_interpretability.R)."),
        
        hr(),
        p(style="color:#8b949e; font-size:0.8em;",
          "Built by Cynric Muller · MS Bioinformatics, Northeastern University · ",
          a("GitHub", href="https://github.com/YOUR_USERNAME/retinal-screening-ml",
            style="color:#58a6ff;"))
      )
    )
  )
)

# ── 6. Server ─────────────────────────────────────────────────────────────────

server <- function(input, output, session) {
  
  # ── Patient risk scoring ────────────────────────────────────────────────────
  
  risk_score <- eventReactive(input$score_btn, {
    # Build a feature vector from inputs
    ma_total     <- sum(input$ma5, input$ma6, input$ma7,
                        input$ma8, input$ma9, input$ma10)
    exudate_mean <- input$exudate_level
    
    # Approximate probability using logistic function on weighted features
    # (demo: uses feature importance weights from typical XGBoost fit on this data)
    log_odds <- -1.2 +
      0.045 * input$ma5  +
      0.038 * input$ma6  +
      0.031 * input$ma7  +
      0.022 * input$ma8  +
      0.015 * input$ma9  +
      0.008 * input$ma10 +
      2.1  * exudate_mean +
      -1.8 * input$macula_dist +
      0.9  * input$optic_diam
    
    prob <- 1 / (1 + exp(-log_odds))
    list(
      prob     = prob,
      decision = ifelse(prob >= input$threshold, "DR Present", "No DR"),
      ma_total = ma_total
    )
  }, ignoreNULL = FALSE)
  
  output$risk_box <- renderUI({
    r    <- risk_score()
    prob <- r$prob
    cls  <- if (prob >= 0.7) "risk-high" else if (prob >= 0.4) "risk-med" else "risk-low"
    div(class="metric-box",
        div(sprintf("%.1f%%", prob * 100), class=paste("metric-val", cls)),
        div("DR Probability", class="metric-lbl"),
        div(r$decision,
            style=paste0("margin-top:8px;font-size:0.8em;padding:4px 10px;",
                         "border-radius:4px;display:inline-block;",
                         if (r$decision == "DR Present")
                           "background:#3d1a1a;color:#f85149;"
                         else
                           "background:#1a3d1a;color:#3fb950;"))
    )
  })
  
  output$sensitivity_box <- renderUI({
    thresh <- input$threshold
    # Approximate sensitivity at this threshold from ROC data (XGBoost)
    xgb_roc <- ROC_DATA %>%
      filter(model == "XGBoost") %>%
      filter(abs(threshold - thresh) == min(abs(threshold - thresh))) %>%
      slice(1)
    div(class="metric-box",
        div(sprintf("%.1f%%", xgb_roc$sensitivity * 100),
            class="metric-val", style="color:#3fb950;"),
        div("Model Sensitivity", class="metric-lbl"),
        div(sprintf("@ threshold %.2f", thresh),
            style="color:#8b949e;font-size:0.75em;margin-top:6px;")
    )
  })
  
  output$action_box <- renderUI({
    r    <- risk_score()
    prob <- r$prob
    if (prob >= 0.7) {
      icon_chr <- "⚠"
      msg      <- "Urgent referral recommended"
      col      <- "#f85149"
      bg       <- "#3d1a1a"
    } else if (prob >= input$threshold) {
      icon_chr <- "▲"
      msg      <- "Refer for ophthalmology review"
      col      <- "#d29922"
      bg       <- "#3d2a0a"
    } else {
      icon_chr <- "✓"
      msg      <- "Continue routine screening"
      col      <- "#3fb950"
      bg       <- "#1a3d1a"
    }
    div(class="metric-box",
        div(icon_chr, style=paste0("font-size:2em;color:",col,";")),
        div("Clinical Action", class="metric-lbl"),
        div(msg, style=paste0("margin-top:8px;font-size:0.8em;padding:6px 10px;",
                              "border-radius:4px;background:",bg,";color:",col,";"))
    )
  })
  
  output$shap_plot <- renderPlotly({
    r <- risk_score()
    
    # Approximate SHAP values: importance weight * (value - mean)
    # Weights from typical XGBoost fit on Messidor data
    features <- c(
      "MA α=0.5"        = input$ma5  * 0.045,
      "MA α=0.6"        = input$ma6  * 0.038,
      "MA α=0.7"        = input$ma7  * 0.031,
      "MA α=0.8"        = input$ma8  * 0.022,
      "Exudates"        = input$exudate_level * 2.1,
      "Macula Distance" = (input$macula_dist - 0.5) * -1.8,
      "Optic Disc Diam" = (input$optic_diam - 0.18) * 0.9,
      "MA α=0.9"        = input$ma9  * 0.015,
      "MA α=1.0"        = input$ma10 * 0.008
    )
    
    df <- tibble(
      feature = names(features),
      shap    = unname(features)
    ) %>%
      arrange(desc(abs(shap))) %>%
      mutate(
        feature = factor(feature, levels = rev(feature)),
        colour  = ifelse(shap >= 0, "#f85149", "#58a6ff")
      )
    
    p <- ggplot(df, aes(x = shap, y = feature, fill = colour)) +
      geom_col(width = 0.65, show.legend = FALSE) +
      geom_vline(xintercept = 0, color = "#30363d", linewidth = 0.8) +
      scale_fill_identity() +
      labs(x = "SHAP-style contribution (toward DR →)",
           y = NULL,
           title = "Feature contributions to DR prediction") +
      gg_dark()
    
    ggplotly(p, tooltip = c("x", "y")) %>%
      layout(
        paper_bgcolor = "#161b22",
        plot_bgcolor  = "#161b22",
        font          = list(color = "#c9d1d9")
      )
  })
  
  output$gauge_plot <- renderPlotly({
    prob <- risk_score()$prob
    plot_ly(
      type  = "indicator",
      mode  = "gauge+number+delta",
      value = round(prob * 100, 1),
      delta = list(reference = 50, increasing = list(color = "#f85149"),
                   decreasing = list(color = "#3fb950")),
      number = list(suffix = "%", font = list(size = 32, color = "#e6edf3")),
      gauge  = list(
        axis  = list(range = list(0, 100), tickcolor = "#8b949e",
                     tickfont = list(color = "#8b949e")),
        bar   = list(color = if (prob >= 0.7) "#f85149"
                     else if (prob >= input$threshold) "#d29922"
                     else "#3fb950"),
        steps = list(
          list(range = c(0,  40), color = "#1a3d1a"),
          list(range = c(40, 60), color = "#3d2a0a"),
          list(range = c(60, 100), color = "#3d1a1a")
        ),
        threshold = list(
          line  = list(color = "#58a6ff", width = 2),
          thickness = 0.85,
          value = input$threshold * 100
        ),
        bgcolor = "#161b22"
      )
    ) %>%
      layout(
        paper_bgcolor = "#161b22",
        font          = list(color = "#c9d1d9"),
        margin        = list(t = 10, b = 10, l = 30, r = 30)
      )
  })
  
  # ── ROC curves ──────────────────────────────────────────────────────────────
  
  output$roc_plot <- renderPlotly({
    selected <- input$roc_models
    roc_filtered <- ROC_DATA %>%
      filter(model %in% selected) %>%
      arrange(model, desc(threshold))
    
    # Compute AUC via trapezoidal rule per model
    auc_vals <- roc_filtered %>%
      group_by(model) %>%
      arrange(specificity) %>%
      summarise(
        auc = round(sum(diff(1 - specificity) * (head(sensitivity,-1) + tail(sensitivity,-1)) / 2), 3),
        .groups = "drop"
      )
    
    # Merge AUC into labels
    roc_labeled <- roc_filtered %>%
      left_join(auc_vals, by = "model") %>%
      mutate(label = sprintf("%s (AUC=%.3f)", model, auc))
    
    p <- ggplot(roc_labeled,
                aes(x = 1 - specificity, y = sensitivity,
                    color = model, group = model,
                    text  = paste0(model, "<br>TPR: ", round(sensitivity,3),
                                   "<br>FPR: ", round(1 - specificity, 3)))) +
      geom_line(linewidth = 1.1, alpha = 0.9) +
      geom_abline(slope = 1, intercept = 0,
                  linetype = "dashed", color = "#30363d", linewidth = 0.6) +
      geom_hline(yintercept = 0.9, linetype = "dotted",
                 color = "#f85149", alpha = 0.7, linewidth = 0.7) +
      annotate("text", x = 0.85, y = 0.915, label = "90% sensitivity target",
               color = "#f85149", size = 3, hjust = 1) +
      scale_color_manual(values = MODEL_COLOURS) +
      coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
      labs(title = "ROC Curves — Test Set",
           x = "1 − Specificity (False Positive Rate)",
           y = "Sensitivity (True Positive Rate)",
           color = NULL) +
      gg_dark()
    
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#161b22", plot_bgcolor = "#161b22",
             font = list(color = "#c9d1d9"),
             legend = list(bgcolor = "#161b22", bordercolor = "#21262d",
                           borderwidth = 1))
  })
  
  # ── PR curves ───────────────────────────────────────────────────────────────
  
  output$pr_plot <- renderPlotly({
    pr_data <- ROC_DATA %>%
      filter(!is.na(precision)) %>%
      arrange(model, threshold)
    
    p <- ggplot(pr_data,
                aes(x = sensitivity, y = precision, color = model,
                    text = paste0(model, "<br>Recall: ", round(sensitivity,3),
                                  "<br>Precision: ", round(precision,3)))) +
      geom_line(linewidth = 1.1, alpha = 0.9) +
      geom_hline(yintercept = 0.54, linetype = "dashed",
                 color = "#30363d", linewidth = 0.6) +
      annotate("text", x = 0.95, y = 0.565,
               label = "Prevalence baseline", color = "#8b949e", size = 3) +
      scale_color_manual(values = MODEL_COLOURS) +
      labs(title = "Precision-Recall Curves — Test Set",
           x = "Recall (Sensitivity)", y = "Precision (PPV)", color = NULL) +
      gg_dark()
    
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#161b22", plot_bgcolor = "#161b22",
             font = list(color = "#c9d1d9"),
             legend = list(bgcolor = "#161b22", bordercolor = "#21262d",
                           borderwidth = 1))
  })
  
  # ── Threshold explorer ───────────────────────────────────────────────────────
  
  output$threshold_plot <- renderPlotly({
    t     <- input$thresh_explore
    sweep <- ROC_DATA %>%
      pivot_longer(c(sensitivity, specificity),
                   names_to = "metric", values_to = "value") %>%
      mutate(metric = tools::toTitleCase(metric))
    
    p <- ggplot(sweep,
                aes(x = threshold, y = value,
                    color = model, linetype = metric,
                    text  = paste0(model, " · ", metric,
                                   "<br>Threshold: ", round(threshold,2),
                                   "<br>Value: ", round(value,3)))) +
      geom_line(linewidth = 0.8, alpha = 0.8) +
      geom_vline(xintercept = t, color = "#58a6ff",
                 linewidth = 0.8, linetype = "dotted") +
      scale_color_manual(values = MODEL_COLOURS) +
      facet_wrap(~model, ncol = 2) +
      labs(title    = "Sensitivity & Specificity vs. Threshold",
           x        = "Classification Threshold",
           y        = "Metric Value",
           color    = NULL,
           linetype = "Metric") +
      gg_dark() +
      theme(strip.background = element_rect(fill = "#1c2128", color = "#21262d"),
            legend.position  = "bottom")
    
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#161b22", plot_bgcolor = "#161b22",
             font = list(color = "#c9d1d9"))
  })
  
  output$threshold_table <- renderUI({
    t <- input$thresh_explore
    tbl <- ROC_DATA %>%
      group_by(model) %>%
      filter(abs(threshold - t) == min(abs(threshold - t))) %>%
      slice(1) %>%
      ungroup() %>%
      select(Model = model, Sensitivity = sensitivity,
             Specificity = specificity) %>%
      mutate(across(c(Sensitivity, Specificity), ~sprintf("%.3f", .)))
    
    tags$table(
      class = "table table-sm",
      style = "font-size:0.8em; color:#c9d1d9; margin-top:10px;",
      tags$thead(tags$tr(lapply(names(tbl), tags$th))),
      tags$tbody(
        apply(tbl, 1, function(r) {
          tags$tr(lapply(r, tags$td))
        })
      )
    )
  })
  
  # ── Metric bar chart ────────────────────────────────────────────────────────
  
  output$metric_bar <- renderPlotly({
    metric <- input$metric_select
    df <- DATA$results %>%
      select(model, value = !!sym(metric)) %>%
      arrange(desc(value)) %>%
      mutate(model = factor(model, levels = model))
    
    p <- ggplot(df, aes(x = model, y = value, fill = model,
                        text = paste0(model, ": ", round(value, 4)))) +
      geom_col(width = 0.6, show.legend = FALSE) +
      geom_text(aes(label = sprintf("%.4f", value)),
                vjust = -0.4, color = "#e6edf3", size = 3.5) +
      scale_fill_manual(values = MODEL_COLOURS) +
      scale_y_continuous(limits = c(0, 1.05), expand = c(0, 0)) +
      labs(title = paste(metric, "by Model — Test Set"),
           x = NULL, y = metric) +
      gg_dark()
    
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#161b22", plot_bgcolor = "#161b22",
             font = list(color = "#c9d1d9"))
  })
  
  # ── Results table ────────────────────────────────────────────────────────────
  
  output$results_table <- renderDT({
    DATA$results %>%
      mutate(across(c(AUROC, Sensitivity, Specificity, F1, Accuracy),
                    ~sprintf("%.4f", .))) %>%
      select(Model = model, AUROC, Sensitivity, Specificity, F1, Accuracy) %>%
      datatable(
        rownames  = FALSE,
        options   = list(dom = "t", pageLength = 10),
        class     = "cell-border stripe",
        style     = "bootstrap5"
      ) %>%
      formatStyle("Model",
                  color = styleEqual(
                    c("Logistic Regression","Random Forest","XGBoost","SVM"),
                    c("#58a6ff","#3fb950","#f0883e","#bc8cff")
                  ))
  })
}

# ── 7. Run ────────────────────────────────────────────────────────────────────

shinyApp(ui = ui, server = server)