suppressMessages(library(rugarch)); suppressMessages(library(jsonlite))
D <- file.path(commandArgs(TRUE)[1], "new_modes")
meta <- fromJSON(file.path(D, "script.json")); out <- list()
for (nm in names(meta)) {
  m <- meta[[nm]]; k <- m$k; th <- setNames(m$params, m$names)
  y <- scan(sprintf("%s/%s_y.csv", D, nm), quiet = TRUE)
  X <- if (k > 0) as.matrix(read.csv(sprintf("%s/%s_X.csv", D, nm), header = FALSE)) else NULL
  Xf <- if (k > 0) as.matrix(read.csv(sprintf("%s/%s_Xf.csv", D, nm), header = FALSE)) else NULL
  nomean <- length(m$fixed) > 0
  spec <- ugarchspec(mean.model = list(armaOrder = c(1, 1), include.mean = !nomean, external.regressors = X),
                     variance.model = list(garchOrder = c(1, 1)), distribution.model = "std")
  fx <- list(ar1 = unname(th["phi1"]), ma1 = unname(th["theta1"]), omega = unname(th["omega"]),
             alpha1 = unname(th["alpha"]), beta1 = unname(th["beta"]), shape = unname(th["nu"]))
  if (!nomean) fx$mu <- unname(th["const"] / (1 - th["phi1"]))
  if (k > 0) for (i in 1:k) fx[[paste0("mxreg", i)]] <- unname(th[paste0("gamma", i)])
  fs <- spec; setfixed(fs) <- fx
  flt <- ugarchfilter(fs, y)
  write.csv(data.frame(res = as.numeric(residuals(flt)), sigma = as.numeric(sigma(flt)), ll = flt@filter$log.likelihoods),
            sprintf("%s/%s_rug_filter.csv", D, nm), row.names = FALSE)
  fc <- ugarchforecast(fs, data = y, n.ahead = 10, external.forecasts = list(mregfor = Xf))
  fit <- ugarchfit(spec, data = y, solver = "hybrid")
  out[[nm]] <- list(coef = as.list(coef(fit)), se_hess = as.list(fit@fit$matcoef[, 2]), LL = likelihood(fit),
                    LL_at_script = likelihood(flt), fc_mean = as.numeric(fitted(fc)), fc_sigma = as.numeric(sigma(fc)))
  cat(sprintf("%-20s rugarch LL=%.4f  LL(at script params)=%.4f  conv=%d\n", nm, likelihood(fit), likelihood(flt), fit@fit$convergence))
}
write_json(out, file.path(D, "rugarch.json"), digits = NA, auto_unbox = TRUE)
