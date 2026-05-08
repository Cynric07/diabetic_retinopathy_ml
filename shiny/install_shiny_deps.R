# install_shiny_deps.R
# Run this once to install all packages needed for the Shiny dashboard

pkgs <- c(
  "shiny",
  "bslib",
  "ggplot2",
  "dplyr",
  "tidyr",
  "plotly",
  "DT",
  "shinyWidgets",
  "shinycssloaders"
)

installed <- rownames(installed.packages())
to_install <- pkgs[!pkgs %in% installed]

if (length(to_install) > 0) {
  cat("Installing:", paste(to_install, collapse = ", "), "\n")
  install.packages(to_install)
} else {
  cat("All Shiny dependencies already installed ✓\n")
}
