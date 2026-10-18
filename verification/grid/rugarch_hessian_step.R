# rugarch's Hessian SEs with its default numDeriv start step (hess.d = 0.1) vs a small step,
# on the high-persistence grid case c03 (alpha + beta ~ 0.993).
suppressMessages(library(rugarch))
D <- file.path(commandArgs(TRUE)[1], "rugarch"); nm <- "c03_arma00_d0_t_X_high-pers"
y <- scan(file.path(D, paste0(nm, "_y.csv")), quiet = TRUE)
X <- as.matrix(read.csv(file.path(D, paste0(nm, "_X.csv")), header = FALSE))
spec <- ugarchspec(mean.model = list(armaOrder = c(0, 0), external.regressors = X),
                   variance.model = list(garchOrder = c(1, 1)), distribution.model = "std")
for (hd in c(0.1, 1e-3)) {
  nc <- list(grad.eps = 1e-4, grad.d = 1e-4, grad.zero.tol = sqrt(.Machine$double.eps / 7e-7),
             hess.eps = 1e-4, hess.d = hd, hess.zero.tol = sqrt(.Machine$double.eps / 7e-7), r = 4, v = 2)
  fit <- ugarchfit(spec, y, solver = "hybrid", numderiv.control = nc)
  cat(sprintf("hess.d=%-6g persistence=%.4f  Hessian SE omega/alpha/beta = %s\n", hd, persistence(fit),
              paste(sprintf("%.5f", fit@fit$matcoef[c("omega", "alpha1", "beta1"), 2]), collapse = " ")))
}
