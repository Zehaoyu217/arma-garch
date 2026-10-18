suppressMessages(library(rugarch))
D <- commandArgs(TRUE)[1]; if (is.na(D)) D <- "cmp"
suppressMessages(library(jsonlite))
meta <- fromJSON(file.path(D, "script_results.json"))
H <- 10
out <- list()
for (nm in names(meta)) {
  m <- meta[[nm]]; p <- m$p; q <- m$q; k <- m$k
  y <- scan(sprintf("%s/%s_y.csv", D, nm), quiet = TRUE)
  X <- if (k > 0) as.matrix(read.csv(sprintf("%s/%s_X.csv", D, nm), header = FALSE)) else NULL
  Xf <- if (k > 0) as.matrix(read.csv(sprintf("%s/%s_Xf.csv", D, nm), header = FALSE)) else NULL
  dist <- if (m$dist == "t") "std" else "norm"
  spec <- ugarchspec(mean.model = list(armaOrder = c(p, q), include.mean = TRUE, external.regressors = X),
                     variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                     distribution.model = dist)
  # ---- map script parameters -> rugarch parametrisation (mu = unconditional mean) ----
  th <- setNames(m$params, m$names)
  phi <- if (p > 0) th[paste0("phi", 1:p)] else numeric(0)
  fx <- list(mu = unname(th["const"] / (1 - sum(phi))), omega = unname(th["omega"]),
             alpha1 = unname(th["alpha"]), beta1 = unname(th["beta"]))
  if (p > 0) for (i in 1:p) fx[[paste0("ar", i)]] <- unname(th[paste0("phi", i)])
  if (q > 0) for (j in 1:q) fx[[paste0("ma", j)]] <- unname(th[paste0("theta", j)])
  if (k > 0) for (i in 1:k) fx[[paste0("mxreg", i)]] <- unname(th[paste0("gamma", i)])
  if (m$dist == "t") fx$shape <- unname(th["nu"])
  fspec <- spec; setfixed(fspec) <- fx
  # ---- (1) filter at the script's MLE ----
  flt <- ugarchfilter(fspec, data = y)
  write.csv(data.frame(res = as.numeric(residuals(flt)), sigma = as.numeric(sigma(flt)),
                       ll = flt@filter$log.likelihoods),
            sprintf("%s/%s_rugarch_filter.csv", D, nm), row.names = FALSE)
  # ---- (2) forecast at the script's MLE ----
  fc <- ugarchforecast(fspec, data = y, n.ahead = H,
                       external.forecasts = if (k > 0) list(mregfor = Xf) else list(mregfor = NULL))
  # ---- (3) free rugarch fit ----
  fit <- ugarchfit(spec, data = y, solver = "hybrid")
  cf <- coef(fit)
  # rugarch loglik at its own MLE, and at the script's MLE (from the filter)
  out[[nm]] <- list(coef = as.list(cf), loglik_fit = likelihood(fit), loglik_at_script = likelihood(flt),
                    se_hess = as.list(fit@fit$matcoef[, 2]), se_rob = as.list(fit@fit$robust.matcoef[, 2]),
                    vcov_hess = vcov(fit), vcov_rob = vcov(fit, robust = TRUE),
                    convergence = fit@fit$convergence,
                    fc_mean = as.numeric(fitted(fc)), fc_sigma = as.numeric(sigma(fc)),
                    hEst_note = "rec.init=all")
  cat(sprintf("%-22s rugarch LL=%.4f  conv=%d  LL(at script MLE)=%.4f\n", nm, likelihood(fit),
              fit@fit$convergence, likelihood(flt)))
}
write_json(out, file.path(D, "rugarch_results.json"), digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
