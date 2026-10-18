# rugarch side of the grid: filter + forecast at the script's estimates, and a free ugarchfit.
suppressMessages(library(rugarch)); suppressMessages(library(jsonlite))
D <- file.path(commandArgs(TRUE)[1], "rugarch")
meta <- fromJSON(file.path(D, "script.json"))
out <- list()
for (nm in names(meta)) {
  m <- meta[[nm]]; p <- m$p; q <- m$q; k <- m$k
  y <- scan(file.path(D, paste0(nm, "_y.csv")), quiet = TRUE)
  X  <- if (k > 0) as.matrix(read.csv(file.path(D, paste0(nm, "_X.csv")), header = FALSE)) else NULL
  Xf <- if (k > 0) as.matrix(read.csv(file.path(D, paste0(nm, "_Xf.csv")), header = FALSE)) else NULL
  spec <- ugarchspec(mean.model = list(armaOrder = c(p, q), include.mean = TRUE, external.regressors = X),
                     variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                     distribution.model = if (m$dist == "t") "std" else "norm")
  th <- setNames(m$params, m$names)
  phi <- if (p > 0) th[paste0("phi", 1:p)] else numeric(0)
  fx <- list(mu = unname(th["const"] / (1 - sum(phi))), omega = unname(th["omega"]),
             alpha1 = unname(th["alpha"]), beta1 = unname(th["beta"]))
  if (p > 0) for (i in 1:p) fx[[paste0("ar", i)]] <- unname(th[paste0("phi", i)])
  if (q > 0) for (j in 1:q) fx[[paste0("ma", j)]] <- unname(th[paste0("theta", j)])
  if (k > 0) for (i in 1:k) fx[[paste0("mxreg", i)]] <- unname(th[paste0("gamma", i)])
  if (m$dist == "t") fx$shape <- unname(th["nu"])
  fs <- spec; setfixed(fs) <- fx
  flt <- ugarchfilter(fs, data = y)
  write.csv(data.frame(res = as.numeric(residuals(flt)), sigma = as.numeric(sigma(flt)),
                       ll = flt@filter$log.likelihoods), file.path(D, paste0(nm, "_rug_filter.csv")), row.names = FALSE)
  fc <- ugarchforecast(fs, data = y, n.ahead = 10, external.forecasts = list(mregfor = Xf))
  fit <- ugarchfit(spec, data = y, solver = "hybrid")
  out[[nm]] <- list(coef = as.list(coef(fit)), vcov_hess = vcov(fit), LL = likelihood(fit),
                    LL_at_script = likelihood(flt), conv = fit@fit$convergence,
                    fc_mean = as.numeric(fitted(fc)), fc_sigma = as.numeric(sigma(fc)))
  cat(sprintf("%-42s rugarch LL=%10.3f conv=%d\n", nm, likelihood(fit), fit@fit$convergence))
}
write_json(out, file.path(D, "rugarch.json"), digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
