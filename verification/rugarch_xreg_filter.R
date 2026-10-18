suppressMessages(library(rugarch)); suppressMessages(library(jsonlite))
D <- commandArgs(TRUE)[1]; if (is.na(D)) D <- "cmp"
R <- fromJSON(file.path(D, "rugarch_results.json")); cf <- unlist(R[["E_arma11_X_t"]]$coef)
y <- scan(file.path(D, "E_arma11_X_t_y.csv"), quiet = TRUE); X <- as.matrix(read.csv(file.path(D, "E_arma11_X_t_X.csv"), header = FALSE))
spec <- ugarchspec(mean.model = list(armaOrder = c(1, 1), external.regressors = X),
                   variance.model = list(garchOrder = c(1, 1)), distribution.model = "std", fixed.pars = as.list(cf))
f <- ugarchfilter(spec, y)
write.csv(data.frame(res = as.numeric(residuals(f)), sigma = as.numeric(sigma(f)), ll = f@filter$log.likelihoods),
          file.path(D, "E_rug_filter_at_rugMLE.csv"), row.names = FALSE)
