#pragma once

void mmat_impl_cpp(int n_row, int n_col, int k, int v, const float *p1,
                   const float *p2, float *res, const float *p3,int block_size, int version);
void mmat_impl_cpp(int n_row, int n_col, int k, int v, const double *p1,
                   const double *p2, double *res, const double *p3, int block_size, int version);
