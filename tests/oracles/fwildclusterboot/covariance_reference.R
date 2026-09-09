cluster_meat_component <- function(score_rows, labels) {
  cluster_sums <- rowsum(score_rows, group = labels, reorder = TRUE)
  cluster_count <- nrow(cluster_sums)
  if (cluster_count <= 1L) {
    stop("Covariance reference requires at least two realised clusters")
  }
  list(
    matrix = (cluster_count / (cluster_count - 1)) * crossprod(cluster_sums),
    cluster_count = cluster_count
  )
}

compute_covariance_reference <- function(
  fit,
  data,
  design_columns,
  restriction_column,
  tolerance = 1e-10
) {
  design <- model.matrix(fit)
  response <- model.response(model.frame(fit))
  bread <- crossprod(design)
  coefficients <- as.numeric(solve(bread, crossprod(design, response)))
  residual <- as.numeric(response - design %*% coefficients)
  score_rows <- design * residual

  rubric <- cluster_meat_component(score_rows, data$rubric)
  arm <- cluster_meat_component(score_rows, data$arm)
  intersection_labels <- interaction(
    data$rubric,
    data$arm,
    drop = TRUE,
    lex.order = TRUE
  )
  intersection <- cluster_meat_component(score_rows, intersection_labels)
  combined <- rubric$matrix + arm$matrix - intersection$matrix

  bread_inverse <- solve(bread)
  raw_covariance <- bread_inverse %*% combined %*% bread_inverse
  symmetrized_covariance <- (raw_covariance + t(raw_covariance)) / 2
  decomposition <- eigen(symmetrized_covariance, symmetric = TRUE)
  clipped_eigenvalues <- ifelse(
    decomposition$values <= tolerance,
    0,
    decomposition$values
  )
  projected_covariance <- decomposition$vectors %*%
    diag(clipped_eigenvalues) %*%
    t(decomposition$vectors)
  projected_covariance <- (projected_covariance + t(projected_covariance)) / 2

  restriction <- matrix(0, nrow = 1, ncol = ncol(design))
  restriction[1, match(restriction_column, design_columns)] <- 1
  contrast <- restriction %*% coefficients
  raw_restriction_covariance <- restriction %*%
    symmetrized_covariance %*%
    t(restriction)
  projected_restriction_covariance <- restriction %*%
    projected_covariance %*%
    t(restriction)
  raw_wald <- as.numeric(t(contrast) %*%
    solve(raw_restriction_covariance, contrast))
  projected_wald <- as.numeric(t(contrast) %*%
    solve(projected_restriction_covariance, contrast))

  list(
    schema_version = 1L,
    row_count = nrow(design),
    parameter_count = ncol(design),
    design_columns = design_columns,
    restriction_column = restriction_column,
    psd_tolerance = tolerance,
    cluster_counts = list(
      rubric = rubric$cluster_count,
      arm = arm$cluster_count,
      intersection = intersection$cluster_count
    ),
    coefficients = coefficients,
    bread = unname(bread),
    meat = list(
      rubric = unname(rubric$matrix),
      arm = unname(arm$matrix),
      intersection = unname(intersection$matrix),
      combined = unname(combined)
    ),
    unprojected_covariance = unname(symmetrized_covariance),
    eigenvalues_before = sort(unname(decomposition$values)),
    eigenvalues_after = sort(unname(clipped_eigenvalues)),
    projected_covariance = unname(projected_covariance),
    materially_indefinite = any(decomposition$values < -tolerance),
    projection_applied = any(clipped_eigenvalues != decomposition$values),
    raw_W_obs = raw_wald,
    projected_W_obs = projected_wald,
    relative_projection_shift = abs(projected_wald - raw_wald) / abs(raw_wald)
  )
}
