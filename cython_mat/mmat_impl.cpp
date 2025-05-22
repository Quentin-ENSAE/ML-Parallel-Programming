#include "mmat_impl.h"
#include <immintrin.h>

#include <algorithm>
#include <iostream>
#include <omp.h>
#include <math.h>
#include <cfloat>


#ifdef __AVX2__
#else
#pragma error "AVX2 not supported"
#endif

// Accès à un élément (i, j)
template <typename DTYPE> inline DTYPE &at(DTYPE *p, int cols, int i, int j) {
  return p[i * cols + j];
}

// Calcul de l'attention par blocs
template <typename DTYPE>
void BlockMatrixMultiply(const DTYPE *A, const DTYPE *B, DTYPE *C, const DTYPE *D, DTYPE *E, int n, int m,
                         int p, int q, int r, int block_size) {
  double dk = sqrt(m);
  double max[n] = {};
  std::fill_n(max, block_size, -DBL_MAX);
  #pragma omp parallel for collapse(2) schedule(dynamic)
  // Tout d'abord on multiplie les deux première matrices
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < p; j += block_size) {
      for (int k = 0; k < m; k += block_size) {
        // Sous-bloc
        for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
          for (int jj = j; jj < std::min(j + block_size, p); ++jj) {
            double sum = 0.0;
            for (int kk = k; kk < std::min(k + block_size, m); ++kk) {
            // On inverse jj et kk pour B, pour transposer la matrice
            // On applique également directement la division par la racine de la dimension
              sum += at(A, m, ii, kk) * at(B, p, jj, kk);
            }
            at(C, p, ii, jj) += sum/dk;
            if(max[ii] < at(C, p, ii, jj)){
                max[ii] = at(C, p, ii, jj);
            }
          }
        }
      }
    }
  }
  //On calcul le soft max
  #pragma omp parallel for schedule(dynamic)
  for(int i = 0; i < n; ++i){
    // On récupère le maximum de chaque ligne
    double exp_sum = 0;
    for(int j = 0; j< p; ++j){
        double exp_val = exp(at(C, p, i, j) - max[i]);
        at(C, p, i, j) = exp_val;
        exp_sum += exp_val;
    }
    // Et on applique le soft max à toutes les valeurs
    for(int j = 0; j< p; ++j){
        at(C, p, i, j) = at(C, p, i, j) / exp_sum;
    }
  }

  //On calcul maintenant la multiplication final de matrice
  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < q; j += block_size) {
      for (int k = 0; k < p; k += block_size) {
        // Sous-bloc
        for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
          for (int jj = j; jj < std::min(j + block_size, q); ++jj) {
            double sum = 0.0;
            for (int kk = k; kk < std::min(k + block_size, p); ++kk) {
              sum += at(C, n, ii, kk) * at(D, q, kk, jj);
            }
            at(E, r, ii, jj) += sum;
          }
        }
      }
    }
  }
}

void mmat_impl_cpp(int n_row, int n_col, int k, int v, int l, const float *p1,
                   const float *p2, float *res, const float *p3, float *res2, int block_size, int version) {
   BlockMatrixMultiply(p1, p2, res, p3, res2, n_row, k, n_col, v, l, block_size);
}

void mmat_impl_cpp(int n_row, int n_col, int k, int v, int l, const double *p1,
                   const double *p2, double *res, const double *p3, double *res2, int block_size, int version) {
   BlockMatrixMultiply(p1, p2, res, p3, res2, n_row, k, n_col, v, l, block_size);
}
