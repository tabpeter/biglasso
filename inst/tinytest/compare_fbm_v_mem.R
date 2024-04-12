# TKP 
# April 2024 
# Objective: explore differences between model fits in-memory and filebacked

devtools::load_all('.')
library(ncvreg)
# load colon data ----------------------------------------------------------------
data(colon)
X <- colon$X |> ncvreg::std()
# X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X)) # cold starts - use more iterations (default is 1000)
y <- colon$y
resid <- drop(y - X %*% init)
X.bm <- as.big.matrix(X)

# take 1 -----------------------------
# file-backed model fit 
fit1 <- biglasso_fit(X = X.bm, y = y, lambda = 0.05, xtx = xtx, r = resid,
                     penalty = "lasso", max.iter = 10000)

# compare with `ncvreg::ncvfit()` -- runs in memory 
fit2 <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid,
               penalty = "lasso", max.iter = 10000)

# test coefficients 
tinytest::expect_equal(fit1$beta, fit2$beta, tolerance = 0.01)

# test residuals
tinytest::expect_equal(fit1$resid, fit2$resid, tolerance = 0.01)

if (interactive()){
  str(fit1) # note: iter = 3773
  str(fit2) # note: iter = 2
  
  nz1 <- which(fit1$beta != 0)
  fit1$beta[nz1]
  
  nz2 <- which(fit2$beta != 0)
  fit2$beta[nz2]
  # note: there are fewer nonzero betas here compared to fit1
  
  if (identical(names(fit1$beta), names(fit2$beta))) {
    fit1$beta[intersect(nz1, nz2)]
    fit2$beta[intersect(nz1, nz2)]
    # note: these estimates are much smaller than fit1! 
  }
  
}

# take 2 -----------------------------
fit1_take2 <- biglasso_fit(X.bm, y, lambda = 0.05, xtx = xtx, r = resid,
                     penalty = "lasso", max.iter = 10000)

# compare with `ncvreg::ncvfit()`
fit2_take2 <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid,
               penalty = "lasso", max.iter = 10000)

# test coefficients 
tinytest::expect_equal(fit1_take2$beta, fit2_take2$beta, tolerance = 0.01)

# test residuals
tinytest::expect_equal(fit1_take2$resid, fit2_take2$resid, tolerance = 0.01)

if (interactive()){
  nz1_take2 <- which(fit1_take2$beta != 0)
  fit1_take2$beta[nz1_take2]
  
  nz2_take2 <- which(fit2_take2$beta != 0)
  fit1_take2$beta[nz2_take2]
  
  if (identical(names(fit1_take2$beta), names(fit2_take2$beta))) {
    fit1_take2$beta[intersect(nz1_take2, nz2_take2)]
    fit2_take2$beta[intersect(nz1_take2, nz2_take2)]
    # now all the estimated coefficients are just close to 0
  }
  
}

# Questions: 
#   (1) why is it that in take 1, the test of residuals passes but the test of beta doesn't?
#   (2) why is it that in take 2, the beta coefficients are different? 
#       The data passed to the functions is the same. Nothing is stochastic here....
#   (3) why does running the same code produce different answers the second time I run it? (Related to issue 2)


