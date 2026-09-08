#!/usr/bin/env Rscript

options(warn = 1)

input_dir <- Sys.getenv("ORACLE_INPUT_DIR", "/input")
output_dir <- Sys.getenv("ORACLE_OUTPUT_DIR", "/output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

contract <- jsonlite::read_json(
  file.path(input_dir, "f6-linear-contract.json"),
  simplifyVector = TRUE
)
data <- read.csv(
  file.path(input_dir, "f6-linear-input.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

design_columns <- contract$design_columns
required_columns <- c("y", "rubric", "arm", design_columns)
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop(paste("Missing oracle input columns:", paste(missing_columns, collapse = ", ")))
}
if (!identical(as.integer(nrow(data)), as.integer(contract$row_count))) {
  stop("Oracle row count does not match the contract")
}

formula <- as.formula(
  paste("y ~ 0 +", paste(sprintf("`%s`", design_columns), collapse = " + "))
)
fit <- lm(formula, data = data)

seed_words <- as.integer(contract$seed_words_signed)
dqrng::dqset.seed(seed_words)
arm_count <- length(unique(data$arm))
weight_matrix <- fwildclusterboot:::get_weights(
  type = "rademacher",
  full_enumeration = FALSE,
  N_G_bootcluster = arm_count,
  boot_iter = as.integer(contract$bootstrap_replicates),
  sampling = "dqrng"
)

# Reset the generator so boottest consumes exactly the persisted schedule.
dqrng::dqset.seed(seed_words)
result <- fwildclusterboot::boottest(
  fit,
  param = contract$restriction_column,
  clustid = c("rubric", "arm"),
  bootcluster = "arm",
  B = as.integer(contract$bootstrap_replicates),
  type = "rademacher",
  impose_null = TRUE,
  engine = "R",
  sampling = "dqrng",
  conf_int = FALSE,
  ssc = fwildclusterboot::boot_ssc(
    adj = FALSE,
    fixef.K = "none",
    cluster.adj = TRUE,
    cluster.df = "conventional"
  )
)

write.csv(
  weight_matrix[, -1, drop = FALSE],
  file.path(output_dir, "f6-linear-rademacher-weights.csv"),
  row.names = FALSE,
  quote = FALSE
)

output <- list(
  schema_version = 1L,
  p_f = as.numeric(result$p_val),
  t_stat = as.numeric(result$t_stat),
  W_obs = as.numeric(result$t_stat)^2,
  bootstrap_replicates = as.integer(result$boot_iter),
  arm_count = arm_count,
  invalid_t_count = as.integer(contract$bootstrap_replicates) - length(result$t_boot),
  r_version = R.version.string,
  fwildclusterboot_version = as.character(utils::packageVersion("fwildclusterboot")),
  dqrng_version = as.character(utils::packageVersion("dqrng")),
  call = list(
    clustid = c("rubric", "arm"),
    bootcluster = "arm",
    B = as.integer(contract$bootstrap_replicates),
    type = "rademacher",
    impose_null = TRUE,
    engine = "R",
    sampling = "dqrng",
    ssc = list(
      adj = FALSE,
      fixef.K = "none",
      cluster.adj = TRUE,
      cluster.df = "conventional"
    )
  )
)
jsonlite::write_json(
  output,
  file.path(output_dir, "f6-linear-r-output.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 17
)
