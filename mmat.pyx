import numpy as np
cimport numpy as np


cdef extern from "mmat_impl.h":
    void mmat_impl_cpp(int n_row, int n_col, int k,
                       const float* p1, const float* p2, float* res,
                       int block_size,
                       int version)
    void mmat_impl_cpp(int n_row, int n_col, int k,
                       const double* p1, const double* p2, double* res,
                       int block_size, int version)


cdef mmat_c_float(const float[:, ::1] a, const float [:, ::1] b, float [:, ::1] res,
                  int block_size, int version):
    mmat_impl_cpp(a.shape[0], b.shape[1], a.shape[0], &a[0, 0], &b[0, 0], &res[0, 0],
                  block_size, version)


cdef mmat_c_double(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] res,
                   int block_size, int version):
    mmat_impl_cpp(a.shape[0], b.shape[1], a.shape[0], &a[0, 0], &b[0, 0], &res[0, 0],
                  block_size, version)


def _mmat(np.ndarray a, np.ndarray b, block_size=16, version=0):
    res = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)

    if a.dtype == np.float32:
        mmat_c_float(a, b, res, block_size, version)
    elif a.dtype == np.float64:
        mmat_c_double(a, b, res, block_size, version)
    else:
        raise NotImplementedError(f"Not implemented for dtype={a.dtype}")
    return res


def mmat(a, b, block_size=16, version=0):
    """Matrix multiplication."""
    assert len(a.shape) == 2 == len(b.shape), (
        f"Only applies on matrices but a.shape={a.shape}, b.shape={b.shape}"
    )
    assert a.shape[1] == b.shape[0], (
        f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"
    )
    assert a.dtype == b.dtype, f"Type mismatch a.dtype={a.dtype}, b.dtype={b.dtype}"
    assert a.flags["C_CONTIGUOUS"], "Matrix a must be contiguous"
    assert b.flags["C_CONTIGUOUS"], "Matrix b must be contiguous"

    return _mmat(a, b, block_size, version)
