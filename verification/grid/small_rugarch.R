# rugarch on the exported small-sample series: AR(1)-GARCH(1,1)-t with 2 external regressors.
suppressMessages(library(rugarch))
D <- commandArgs(TRUE)[1]
for (T in c(120, 180)) {
  Y <- as.matrix(read.csv(file.path(D, sprintf("T%d_y.csv", T)), header = FALSE))
  X <- as.matrix(read.csv(file.path(D, sprintf("T%d_X.csv", T)), header = FALSE))
  spec <- ugarchspec(mean.model = list(armaOrder = c(1, 0), include.mean = TRUE, external.regressors = X),
                     variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), distribution.model = "std")
  out <- t(sapply(seq_len(ncol(Y)), function(i) {
    fit <- tryCatch(ugarchfit(spec, Y[, i], solver = "hybrid"), error = function(e) NULL, warning = function(w) NULL)
    if (is.null(fit) || fit@fit$convergence != 0) return(rep(NA, 9))
    c(coef(fit)[c("mu", "ar1", "mxreg1", "mxreg2", "omega", "alpha1", "beta1", "shape")], fit@fit$convergence)
  }))
  colnames(out) <- c("mu", "ar1", "mxreg1", "mxreg2", "omega", "alpha1", "beta1", "shape", "conv")
  write.csv(out, file.path(D, sprintf("T%d_rugarch.csv", T)), row.names = FALSE)
}
