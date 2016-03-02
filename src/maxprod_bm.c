#include <math.h>
#include <string.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"
#include <R.h>
#include <R_ext/Applic.h>

// c++ functions for big.matrix
int get_row_bm(SEXP xP);
int get_col_bm(SEXP xP);
double crossprod_bm(SEXP xP, double *y_, int *row_idx_, double center_, double scale_, int n_row, int j);

SEXP maxprod_bm(SEXP X_, SEXP y_, SEXP row_idx_, SEXP center_, SEXP scale_, SEXP v_, SEXP m_) {

  // Declarations
  int n = length(row_idx_);
  int p = length(v_);

  SEXP z;
  PROTECT(z = allocVector(REALSXP, 1));
  REAL(z)[0] = 0;
  double zz;

  double *y = REAL(y_);
  double *m = REAL(m_);
  int *v = INTEGER(v_);
  int *row_idx = INTEGER(row_idx_);
  double *center = REAL(center_);
  double *scale = REAL(scale_);
  
  for (int j=0; j<p; j++) {
    zz = crossprod_bm(X_, y, row_idx, center[j], scale[j], n, v[j]-1) / m[v[j]-1];
    if (fabs(zz) > REAL(z)[0]) REAL(z)[0] = fabs(zz);
  }

  // Return list
  UNPROTECT(1);
  return(z);
}
