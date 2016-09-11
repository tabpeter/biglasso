
#include <RcppArmadillo.h>
#include <iostream>
#include "bigmemory/BigMatrix.h"
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"
#include "bigmemory/isna.hpp"
#include <omp.h>
#include <stdlib.h>

//#include "defines.h"

using namespace Rcpp;
using namespace std;

// Cross product of y with jth column of X
double crossprod(double *X, double *y, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += X[nn+i]*y[i];
  return(val);
}

int sum_discard(int *discards, int p) {
  int sum  = 0;
  for (int j = 0; j < p; j++) {
    sum += discards[j];
  }
  return sum;
}

int sum_int(int *vec, int p) {
  int j, sum  = 0;
  for (j = 0; j < p; j++) {
    sum += vec[j];
  }
  return sum;
}

double sign(double x) {
  if(x>0.00000000001) return 1.0;
  else if(x<-0.00000000001) return -1.0;
  else return 0.0;
}

double sum(double *x, int n) {
  double val=0;
  for (int i=0;i<n;i++) val += x[i];
  return(val);
}

// Sum of squares of jth column of X
double sqsum(double *X, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += pow(X[nn+i], 2);
  return(val);
}

double lasso(double z, double l1, double l2, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else return(s*(fabs(z)-l1)/(v*(1+l2)));
}

// Gaussian loss
double gLoss(double *r, int n) {
  double l = 0;
  for (int i=0;i<n;i++) l = l + pow(r[i],2);
  return(l);
}

int checkConvergence(arma::sp_mat beta, double *beta_old, double eps, int l, int p) {
  int converged = 1;
  for (int j = 0; j < p; j++) {
    if (fabs((beta(j, l) - beta_old[j]) / beta_old[j]) > eps) {
      converged = 0;
      break;
    }
  }
  return(converged);
}

// get X[i, j]: i-th row, j-th column element
template<typename T>
double get_elem_bm(XPtr<BigMatrix> xpMat, double center_, double scale_, int i, int j) {
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol = xAcc[j];
  double res = (xCol[i] - center_) / scale_;
  return res;
}

//crossprod - given specific rows of X
template<typename T>
double crossprod_bm(XPtr<BigMatrix> xpMat, double *y_, int *row_idx_, double center_, 
                               double scale_, int n_row, int j) {
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol = xAcc[j];

  double sum = 0.0;
  double sum_xy = 0.0;
  double sum_y = 0.0;
  for (int i=0; i < n_row; i++) {
    // row_idx only used by xP, not by y;
    sum_xy = sum_xy + xCol[row_idx_[i]] * y_[i];
    sum_y = sum_y + y_[i];
  }
  sum = (sum_xy - center_ * sum_y) / scale_;

  return sum;
}

//crossprod_resid - given specific rows of X: separate computation
template<typename T>
double crossprod_resid(XPtr<BigMatrix> xpMat, double *y_, double sumY_, int *row_idx_, 
                                  double center_, double scale_, int n_row, int j) {
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol = xAcc[j];
  
  double sum = 0.0;
  for (int i=0; i < n_row; i++) {
    sum = sum + xCol[row_idx_[i]] * y_[i];
  }
  sum = (sum - center_ * sumY_) / scale_;
  return sum;
}

// update residul vector if variable j enters eligible set
template<typename T>
void update_resid(XPtr<BigMatrix> xpMat, double *r, double shift, int *row_idx_, 
                               double center_, double scale_, int n_row, int j) {
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol = xAcc[j];
  
  for (int i=0; i < n_row; i++) {
    r[i] -= shift * (xCol[row_idx_[i]] - center_) / scale_;
  }
}

// // Sum of squares of jth column of X
// double sqsum_bm(SEXP xP, int n_row, int j, int useCores) {
//   XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
//   MatrixAccessor<double> xAcc(*xpMat);
//   double *xCol = xAcc[j];
//   
// //   omp_set_dynamic(0);
// //   omp_set_num_threads(useCores);
//   //Rprintf("sqsum_bm: Number of used threads: %d\n", omp_get_num_threads());
//   double val = 0.0;
//   // #pragma omp parallel for reduction(+:val)
//   for (int i=0; i < n_row; i++) {
//     val += pow(xCol[i], 2);
// //     if (i == 0) {
// //       Rprintf("sqsum_bm: Number of used threads: %d\n", omp_get_num_threads());
// //     }
//   }
//   return val;
// }

// Weighted sum of residuals
double wsum(double *r, double *w, int n_row) {
  double val = 0.0;
  for (int i = 0; i < n_row; i++) {
    val += r[i] * w[i];
  }
  return val;
}

// Weighted cross product of y with jth column of x
template<typename T>
double wcrossprod_resid(XPtr<BigMatrix> xpMat, double *y, double sumYW_, int *row_idx_, 
                        double center_, double scale_, double *w, int n_row, int j) {
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol = xAcc[j];

  double val = 0.0;
  for (int i = 0; i < n_row; i++) {
    val += xCol[row_idx_[i]] * y[i] * w[i];
  }
  val = (val - center_ * sumYW_) / scale_;
  
  return val;
}


// Weighted sum of squares of jth column of X
// sum w_i * x_i ^2 = sum w_i * ((x_i - c) / s) ^ 2
// = 1/s^2 * (sum w_i * x_i^2 - 2 * c * sum w_i x_i + c^2 sum w_i)
template<typename T>
double wsqsum_bm(XPtr<BigMatrix> xpMat, double *w, int *row_idx_, double center_, 
                            double scale_, int n_row, int j) {
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol = xAcc[j];
  
  double val = 0.0;
  double sum_wx_sq = 0.0;
  double sum_wx = 0.0;
  double sum_w = 0.0;
  for (int i = 0; i < n_row; i++) {
    sum_wx_sq += w[i] * pow(xCol[row_idx_[i]], 2);
    sum_wx += w[i] * xCol[row_idx_[i]];
    sum_w += w[i];
  }
  val = (sum_wx_sq - 2 * center_ * sum_wx + pow(center_, 2) * sum_w) / pow(scale_, 2);
  return val;
}

// standardize
template<typename T>
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                  vector<double> &z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, double *y, 
                                  int *row_idx, double lambda_min, double alpha, int n, int p) {
  MatrixAccessor<T> xAcc(*xMat);
  T *xCol;
  double sum_xy, sum_y;
  double zmax = 0.0, zj = 0.0;
  int i, j;
  
  for (j = 0; j < p; j++) {
    xCol = xAcc[j];
    sum_xy = 0.0;
    sum_y = 0.0;
    
    for (i = 0; i < n; i++) {
      center[j] += xCol[row_idx[i]];
      scale[j] += pow(xCol[row_idx[i]], 2);
      
      sum_xy = sum_xy + xCol[row_idx[i]] * y[i];
      sum_y = sum_y + y[i];
    }
    
    center[j] = center[j] / n; //center
    scale[j] = sqrt(scale[j] / n - pow(center[j], 2)); //scale
    
    if (scale[j] > 1e-6) {
      col_idx.push_back(j);
      zj = (sum_xy - center[j] * sum_y) / (scale[j] * n); //residual
      if (fabs(zj) > zmax) {
        zmax = fabs(zj);
        *xmax_ptr = j;
      }
      z.push_back(zj);
    }
  }
  *p_keep_ptr = col_idx.size();
  *lambda_max_ptr = zmax / alpha;
}

void free_memo_hsr(double *a, double *r, int *e1, int *e2) {
  free(a);
  free(r);
  free(e1);
  free(e2);
}

// -----------------------------------------------------------------------------
// Following functions are used for EDPP rule
// -----------------------------------------------------------------------------
// 
// // apply EDPP - by chunking reading, openmp
// void edpp_screen_by_chunk_omp(int *discard_beta, const char *xf_bin, int nchunks, int chunk_cols, 
//                           double *o, int *row_idx, NumericVector &center, 
//                           NumericVector &scale, int n, int p, double rhs, int n_total) {
//   unsigned long int chunk_size = chunk_cols * sizeof(double) * n_total;
//   ifstream xfile(xf_bin, ios::in|ios::binary);
//   
//   int i, j;
//   double lhs;
//   double sum_xy;
//   double sum_y;
//   
//   if (xfile.is_open()) {
//     char *memblock_x;
//     int chunk_i;
//     //int chunks = 0;
//     int col_pos;
//     int col_j;
//     double *X;
// 
//     memblock_x = (char *) calloc(chunk_size, sizeof(char));
//     for (chunk_i = 0; chunk_i < nchunks; chunk_i++) {
//       col_pos = chunk_i * chunk_cols; // current column position offset
//       
//       xfile.seekg (chunk_i * chunk_size, ios::beg);
//       xfile.read (memblock_x, chunk_size);
//       X = (double*) memblock_x;//reinterpret as doubles
//       
//       // loop over loaded columns and do crossprod
//       #pragma omp parallel for private(sum_xy, sum_y, j, lhs, col_j) default(shared) schedule(static) 
//       for (j = 0; j < chunk_cols; j++) {
//         sum_xy = 0.0;
//         sum_y = 0.0;
//         col_j = col_pos + j; //absolute position of current column j
//         for (i = 0; i < n; i++) {
//           sum_xy = sum_xy + X[j*n_total+row_idx[i]] * o[i];
//           sum_y = sum_y + o[i];
//         }
//         //printf("\nresult[%d] = %f\n\n", j, result[col_pos + j]);
//         lhs = fabs((sum_xy - center[col_j] * sum_y) / scale[col_j]);
//         if (lhs < rhs) {
//           discard_beta[col_j] = 1;
//         } else {
//           discard_beta[col_j] = 0;
//         }
//       }
//     }
//     free(memblock_x);
//     xfile.close();
//   } else {
//     Rprintf("Open file failed! filename = %s, chunk_size = %lu\n", xf_bin, chunk_size);
//     Rcpp::stop("Open file failed!");
//     // exit(EXIT_FAILURE);
//   }
// }
// 
// 
// // apply EDPP - by chunking reading, sequential
// void edpp_screen_by_chunk(int *discard_beta, const char *xf_bin, int nchunks, int chunk_cols, 
//                           double *o, int *row_idx, NumericVector &center, 
//                           NumericVector &scale, int n, int p, double rhs, int n_total) {
//   unsigned long int chunk_size = chunk_cols * sizeof(double) * n_total;
//   ifstream xfile(xf_bin, ios::in|ios::binary);
//   
//   int i, j;
//   double lhs;
//   double sum_xy;
//   double sum_y;
//   
//   if (xfile.is_open()) {
//     //Rprintf("Open file succeeded! filename = %s, chunk_size = %lu\n", xf_bin, chunk_size);
//     streampos size_x;
//     char *memblock_x;
//     int chunk_i = 0;
//     //int chunks = 0;
//     int col_pos = 0;
//     int col_j = 0;
//     
//     //xfile.seekg (0, ios::end);
//     //size_x = xfile.tellg();
//     //xfile.seekg (0, ios::beg);
//     //cout << "\tSize_x = " << size_x << endl;
//     // I need guarantee this is dividable;
//     //chunks = size_x / chunk_size;
//    
//     memblock_x = (char *) calloc(chunk_size, sizeof(char));
//     while (chunk_i < nchunks) {
//       //printf("\tChunk %d\n", chunk_i);
//       col_pos = chunk_i * chunk_cols; // current column position offset
//       xfile.seekg (chunk_i * chunk_size, ios::beg);
//       xfile.read (memblock_x, chunk_size);
//       
//       //size_t count = xfile.gcount();
//       // If nothing has been read, break
//       //if (!count) {
//       //  Rprintf("\n\t Error in reading chunk %d\n", chunk_i);
//       //  break;
//       //} 
//       double *X = (double*) memblock_x;//reinterpret as doubles
//       
//       // loop over loaded columns and do crossprod
//       for (j = 0; j < chunk_cols; j++) {
//         sum_xy = 0.0;
//         sum_y = 0.0;
//         col_j = col_pos + j; //absolute position of current column j
//         for (i = 0; i < n; i++) {
//           sum_xy = sum_xy + X[j*n_total+row_idx[i]] * o[i];
//           sum_y = sum_y + o[i];
//         }
//         //printf("\nresult[%d] = %f\n\n", j, result[col_pos + j]);
//         lhs = fabs((sum_xy - center[col_j] * sum_y) / scale[col_j]);
//         if (lhs < rhs) {
//           discard_beta[col_j] = 1;
//         } else {
//           discard_beta[col_j] = 0;
//         }
//       }
//       chunk_i++;
//       
//     }
//     free(memblock_x);
//     xfile.close();
//   } else {
//     Rprintf("Open file failed! filename = %s, chunk_size = %lu\n", xf_bin, chunk_size);
//     Rcpp::stop("Open file failed!");
//     // exit (EXIT_FAILURE);
//   }
// }
// 
// void edpp_screen2(int *discard_beta, SEXP xP, double *o, int *row_idx, 
//                   NumericVector &center, NumericVector &scale, int n, int p, 
//                   double rhs) {
//   XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
//   MatrixAccessor<double> xAcc(*xpMat);
//   
//   int j;
//   double lhs;
//   double sum_xy;
//   double sum_y;
// 
//   #pragma omp parallel for private(j, lhs, sum_xy, sum_y) default(shared) schedule(static) 
//   for (j = 0; j < p; j++) {
//     sum_xy = std::inner_product(xAcc[j], xAcc[j]+n, o, 0.0);
//     sum_y = std::accumulate(o, o+n, 0.0);
//     lhs = fabs((sum_xy - center[j] * sum_y) / scale[j]);
//     if (lhs < rhs) {
//       discard_beta[j] = 1;
//     } else {
//       discard_beta[j] = 0;
//     }
//   }
// }



// -----------------------------------------------------------------------------
// Following functions are directly callled inside R
// -----------------------------------------------------------------------------

// standardize big matrix, return just 'center' and 'scale'
RcppExport SEXP standardize_bm(SEXP xP, SEXP row_idx_) {
  BEGIN_RCPP
  SEXP __sexp_result;
  {
    Rcpp::RNGScope __rngScope;
    XPtr<BigMatrix> xMat(xP);
    MatrixAccessor<double> xAcc(*xMat);
    int ncol = xMat->ncol();
    
    NumericVector c(ncol);
    NumericVector s(ncol);
    IntegerVector row_idx(row_idx_);
    int nrow = row_idx.size();
    
    for (int j = 0; j < ncol; j++) {
      c[j] = 0; //center
      s[j] = 0; //scale
      for (int i = 0; i < nrow; i++) {
        c[j] += xAcc[j][row_idx[i]]; 
        s[j] += pow(xAcc[j][row_idx[i]], 2);
      }
      c[j] = c[j] / nrow;
      s[j] = sqrt(s[j] / nrow - pow(c[j], 2));
    }
    PROTECT(__sexp_result = Rcpp::List::create(c, s));
  }
  UNPROTECT(1);
  return __sexp_result;
  END_RCPP
}

// compute eta = X %*% beta. X: n-by-p; beta: p-by-l. l is length of lambda
RcppExport SEXP get_eta(SEXP xP, SEXP row_idx_, SEXP beta, SEXP idx_p, SEXP idx_l) {
  BEGIN_RCPP
  SEXP __sexp_result;
  {
    Rcpp::RNGScope __rngScope;
    XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
    MatrixAccessor<double> xAcc(*xpMat);
    
    // sparse matrix for beta: only pass the non-zero entries and their indices;
    arma::sp_mat sp_beta = Rcpp::as<arma::sp_mat>(beta);
    
    IntegerVector row_idx(row_idx_);
    IntegerVector index_p(idx_p);
    IntegerVector index_l(idx_l);
    
    int n = row_idx.size();
    int l = sp_beta.n_cols;
    int nnz = index_p.size();
    
    // initialize result
    arma::sp_mat sp_res = arma::sp_mat(n, l);
    
    for (int k = 0; k < nnz; k++) {
      for (int i = 0; i < n; i++) {
        //double x = (xAcc[index_p[k]][row_idx[i]] - center[index_p[k]]) / scale[index_p[k]];
        // NOTE: beta here is unstandardized; so no need to standardize x
        double x = xAcc[index_p[k]][row_idx[i]];
        sp_res(i, index_l[k]) += x * sp_beta(index_p[k], index_l[k]);
      }
    }
    
    PROTECT(__sexp_result = Rcpp::wrap(sp_res));
  }
  UNPROTECT(1);
  return __sexp_result;
  END_RCPP
}


