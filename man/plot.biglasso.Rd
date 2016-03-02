\name{plot.biglasso}

\alias{plot.biglasso}

\title{Plot coefficients from a "biglasso" object}

\description{
Produce a plot of the coefficient paths for a fitted \code{\link{biglasso}} object.
}

\usage{
\method{plot}{biglasso}(x, alpha = 1, log.l = TRUE, ...)
}

\arguments{
  \item{x}{Fitted \code{"biglasso"} model.}
  \item{alpha}{Controls alpha-blending, helpful when the number of
    covariates is large.  Default is alpha=1.}
  \item{log.l}{Should horizontal axis be on the log scale?  Default is
    TRUE.} 
  \item{\dots}{Other graphical parameters to \code{plot}}
}

\author{
  Yaohui Zeng and Patrick Breheny
  
  Maintainer: Yaohui Zeng <yaohui-zeng@uiowa.edu>
}

\seealso{\code{\link{biglasso}}, \code{\link{cv.biglasso}}}

\examples{
## See examples in "biglasso"
}

\keyword{models}
\keyword{regression}
