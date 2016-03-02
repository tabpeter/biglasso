\name{biglasso-internal}

\title{Internal biglasso functions}
\alias{setupLambda}
\alias{loss}

\description{Internal biglasso functions}

\usage{
setupLambda(X, y, row.idx, center, scale, family, alpha, 
            lambda.min, nlambda, penalty.factor)

loss.biglasso(y,yhat,family)
}

\author{
  Yaohui Zeng and Patrick Breheny
  
  Maintainer: Yaohui Zeng <yaohui-zeng@uiowa.edu>
}

\details{
These are not intended for use by users. \code{setupLambda} creates an 
appropriate vector of regularization parameter values.  \code{loss.biglasso} 
calculates the value of the loss function for the given predictions (used for
cross-validation).
}

\keyword{internal}
