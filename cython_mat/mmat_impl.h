#pragma once

void mmat_impl_cpp(int n_row, int n_col, int k, int v, int l, const float *p1,
                   const float *p2, float *res, const float *p3, float *res2, int block_size, int version);
void mmat_impl_cpp(int n_row, int n_col, int k, int v, int l, const double *p1,
                   const double *p2, double *res, const double *p3, double *res2, int block_size, int version);
