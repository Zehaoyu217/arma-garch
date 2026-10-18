suppressMessages(library(rugarch))
D <- commandArgs(TRUE)[1]; if (is.na(D)) D <- "cmp"
y <- scan(file.path(D, "A_arma11_t_y.csv"), quiet = TRUE); n <- length(y)
spec <- ugarchspec(mean.model = list(armaOrder = c(1, 1)), variance.model = list(garchOrder = c(1, 1)),
                   distribution.model = "std")
fit <- ugarchfit(spec, data = y, solver = "hybrid")
H <- fit@fit$hessian; S <- fit@fit$scores; th <- coef(fit)
se <- function(V) round(sqrt(diag(V)), 4)
cat("rugarch reported robust SE       :", se(fit@fit$robust.cvar), "\n")
cat("H^-1 S'S H^-1 from fit@fit$scores:", se(solve(H) %*% crossprod(S) %*% solve(H)), "\n")
cat("n H^-1 cov(S) H^-1 (robustvcv eq):", se(n * solve(H) %*% cov(S) %*% solve(H)), "\n")
# scores re-derived exactly as robustvcv does (h = max(|theta| eps^(1/3), 1e-9)), via ugarchfilter
llt <- function(p) { sp <- spec; setfixed(sp) <- as.list(p); ugarchfilter(sp, y)@filter$log.likelihoods }
h <- pmax(abs(th * .Machine$double.eps^(1/3)), 1e-9)
S2 <- sapply(seq_along(th), function(j) { e <- replace(numeric(length(th)), j, h[j]); (llt(th + e) - llt(th - e)) / (2 * h[j]) })
cat("n H^-1 cov(S2) H^-1, S2 via filter:", se(n * solve(H) %*% cov(S2) %*% solve(H)), "\n")
cat("max |S2 - (-S)| :", max(abs(S2 + S)), "  max |S2 - S| :", max(abs(S2 - S)), "\n")
cat("rugarch Hessian SE (matcoef)     :", round(fit@fit$matcoef[, 2], 4), "\n")
cat("sqrt diag solve(H)               :", se(solve(H)), "\n")
cat("cvar == solve(H)?  max|diff| =", max(abs(fit@fit$cvar - solve(H))), "\n")
