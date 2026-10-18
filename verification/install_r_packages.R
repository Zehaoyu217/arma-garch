options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 8)
pk <- c("rugarch", "fGarch", "numDeriv", "jsonlite")
install.packages(pk)
for (p in pk) cat(p, as.character(packageVersion(p)), "\n")
