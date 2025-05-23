# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport exp, sqrt
from libc.stdlib cimport malloc, free

# Include our AVX header
cdef extern from "attention_avx.h":
    void avx_dot_product_add_double(const double* a, const double* b, double* result, int length) nogil
    void avx_matrix_block_mul_double(const double* Q, const double* K, double* QKt, 
                                    int batch_idx, int i_start, int i_end, 
                                    int j_start, int j_end, int d_model, int seq_len) nogil
    void avx_attention_output_block(const double* attention_weights, const double* V, double* output,
                                   int batch_idx, int i_start, int i_end,
                                   int d_start, int d_end, int seq_len, int d_model) nogil

cdef inline double min_c(double a, double b) nogil:
    return a if a < b else b

def cython_block_attention(np.ndarray[np.float64_t, ndim=3] Q, 
                          np.ndarray[np.float64_t, ndim=3] K, 
                          np.ndarray[np.float64_t, ndim=3] V, 
                          int block_size=16):
    """
    Cython implementation of the attention mechanism with block-based parallelization
    
    Parameters:
    -----------
    Q : ndarray of shape (batch_size, seq_len, d_model)
        Query matrix
    K : ndarray of shape (batch_size, seq_len, d_model)
        Key matrix
    V : ndarray of shape (batch_size, seq_len, d_model)
        Value matrix
    block_size : int
        Size of blocks for parallelization
        
    Returns:
    --------
    output : ndarray of shape (batch_size, seq_len, d_model)
        Result of the attention operation
    """
    cdef int batch_size = Q.shape[0]
    cdef int seq_len = Q.shape[1]
    cdef int d_model = Q.shape[2]
    cdef double d_k_sqrt = sqrt(K.shape[2])
    
    # Define arrays
    cdef np.ndarray[np.float64_t, ndim=3] QKt = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] attention_weights = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((batch_size, seq_len, d_model), dtype=np.float64)
    
    cdef int i, j, b, i_block, j_block, d, i_end, j_end, d_end
    cdef double max_val, sum_exp
    
    # Step 1: Compute QK^T / sqrt(d_k) in blocks
    for b in range(batch_size):
        for i_block in range(0, seq_len, block_size):
            i_end = min(i_block + block_size, seq_len)
            for j_block in range(0, seq_len, block_size):
                j_end = min(j_block + block_size, seq_len)
                
                # Process this block
                for i in range(i_block, i_end):
                    for j in range(j_block, j_end):
                        # Use numpy's dot product for this part
                        QKt[b, i, j] = 0
                        for d in range(d_model):
                            QKt[b, i, j] += Q[b, i, d] * K[b, j, d]
                        QKt[b, i, j] /= d_k_sqrt
    
    # Step 2: Apply softmax to each row
    for b in range(batch_size):
        for i in range(seq_len):
            # Find max for numerical stability
            max_val = QKt[b, i, 0]
            for j in range(1, seq_len):
                if QKt[b, i, j] > max_val:
                    max_val = QKt[b, i, j]
            
            # Compute exp and sum
            sum_exp = 0.0
            for j in range(seq_len):
                attention_weights[b, i, j] = exp(QKt[b, i, j] - max_val)
                sum_exp += attention_weights[b, i, j]
            
            # Normalize
            for j in range(seq_len):
                attention_weights[b, i, j] /= sum_exp
    
    # Step 3: Compute the final output (attention_weights @ V) in blocks
    for b in range(batch_size):
        for i in range(seq_len):
            for d in range(d_model):
                output[b, i, d] = 0
                for j in range(seq_len):
                    output[b, i, d] += attention_weights[b, i, j] * V[b, j, d]
    
    return output


def cython_optimized_attention(np.ndarray[np.float64_t, ndim=3] Q, 
                              np.ndarray[np.float64_t, ndim=3] K, 
                              np.ndarray[np.float64_t, ndim=3] V,
                              int block_size=16):
    """
    Optimized Cython implementation of the attention mechanism with block-based parallelization
    using pure C++ operations instead of numpy
    
    Parameters:
    -----------
    Q : ndarray of shape (batch_size, seq_len, d_model)
        Query matrix
    K : ndarray of shape (batch_size, seq_len, d_model)
        Key matrix
    V : ndarray of shape (batch_size, seq_len, d_model)
        Value matrix
    block_size : int
        Size of blocks for parallelization
        
    Returns:
    --------
    output : ndarray of shape (batch_size, seq_len, d_model)
        Result of the attention operation
    """
    cdef int batch_size = Q.shape[0]
    cdef int seq_len = Q.shape[1]
    cdef int d_model = Q.shape[2]
    cdef double d_k_sqrt = sqrt(K.shape[2])
    
    # Define arrays
    cdef np.ndarray[np.float64_t, ndim=3] QKt = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] attention_weights = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((batch_size, seq_len, d_model), dtype=np.float64)
    
    cdef int b, i, j, k, d
    cdef int i_block, j_block, k_block, i_end, j_end, k_end
    cdef double max_val, sum_exp

    # Step 1: Compute QK^T / sqrt(d_k) in blocks
    # We can't use min() or range() in nogil context, so we'll do this part with gil
    for b in range(batch_size):
        for i_block in range(0, seq_len, block_size):
            i_end = min(i_block + block_size, seq_len)
            for j_block in range(0, seq_len, block_size):
                j_end = min(j_block + block_size, seq_len)
                
                # Process this block with tiling for better cache efficiency
                for k_block in range(0, d_model, block_size):
                    k_end = min(k_block + block_size, d_model)
                    
                    # Compute partial dot products for this block
                    for i in range(i_block, i_end):
                        for j in range(j_block, j_end):
                            for k in range(k_block, k_end):
                                QKt[b, i, j] += Q[b, i, k] * K[b, j, k]
    
    # Scale by sqrt(d_k)
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                QKt[b, i, j] /= d_k_sqrt
    
    # Step 2: Apply softmax to each row with parallelization
    # Since finding max and sum are associative operations, we can safely parallelize this
    with nogil, parallel():
        for b in prange(batch_size):
            for i in range(seq_len):
                # Find max for numerical stability
                max_val = QKt[b, i, 0]
                for j in range(1, seq_len):
                    if QKt[b, i, j] > max_val:
                        max_val = QKt[b, i, j]
                
                # Compute exp and sum
                sum_exp = 0.0
                for j in range(seq_len):
                    attention_weights[b, i, j] = exp(QKt[b, i, j] - max_val)
                    sum_exp += attention_weights[b, i, j]
                
                # Normalize
                for j in range(seq_len):
                    attention_weights[b, i, j] /= sum_exp
    
    # Step 3: Compute the final output (attention_weights @ V) in blocks
    # We'll do this part with gil since we need to use Python operations
    for b in range(batch_size):
        for i_block in range(0, seq_len, block_size):
            i_end = min(i_block + block_size, seq_len)
            for d_block in range(0, d_model, block_size):
                d_end = min(d_block + block_size, d_model)
                
                # Process this block with tiling
                for j_block in range(0, seq_len, block_size):
                    j_end = min(j_block + block_size, seq_len)
                    
                    for i in range(i_block, i_end):
                        for d in range(d_block, d_end):
                            for j in range(j_block, j_end):
                                output[b, i, d] += attention_weights[b, i, j] * V[b, j, d]
    
    return output 

def cython_avx_attention(np.ndarray[np.float64_t, ndim=3] Q, 
                        np.ndarray[np.float64_t, ndim=3] K, 
                        np.ndarray[np.float64_t, ndim=3] V,
                        int block_size=16):
    """
    AVX-optimized Cython implementation of the attention mechanism
    
    Parameters:
    -----------
    Q : ndarray of shape (batch_size, seq_len, d_model)
        Query matrix
    K : ndarray of shape (batch_size, seq_len, d_model)
        Key matrix
    V : ndarray of shape (batch_size, seq_len, d_model)
        Value matrix
    block_size : int
        Size of blocks for parallelization
        
    Returns:
    --------
    output : ndarray of shape (batch_size, seq_len, d_model)
        Result of the attention operation
    """
    cdef int batch_size = Q.shape[0]
    cdef int seq_len = Q.shape[1]
    cdef int d_model = Q.shape[2]
    cdef double d_k_sqrt = sqrt(K.shape[2])
    
    # Define arrays
    cdef np.ndarray[np.float64_t, ndim=3] QKt = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] attention_weights = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((batch_size, seq_len, d_model), dtype=np.float64)
    
    cdef int b, i, j, k, d
    cdef int i_block, j_block, k_block, i_end, j_end, k_end, d_end
    cdef double max_val, sum_exp
    
    # Get pointers to the raw data
    cdef double* Q_ptr = <double*>Q.data
    cdef double* K_ptr = <double*>K.data
    cdef double* V_ptr = <double*>V.data
    cdef double* QKt_ptr = <double*>QKt.data
    cdef double* attention_weights_ptr = <double*>attention_weights.data
    cdef double* output_ptr = <double*>output.data
    
    # Step 1: Compute QK^T / sqrt(d_k) in blocks using AVX
    for b in range(batch_size):
        for i_block in range(0, seq_len, block_size):
            i_end = min(i_block + block_size, seq_len)
            for j_block in range(0, seq_len, block_size):
                j_end = min(j_block + block_size, seq_len)
                
                # Use our optimized AVX function for this block
                avx_matrix_block_mul_double(
                    Q_ptr, K_ptr, QKt_ptr,
                    b, i_block, i_end, j_block, j_end, 
                    d_model, seq_len
                )
    
    # Scale by sqrt(d_k)
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                QKt[b, i, j] /= d_k_sqrt
    
    # Step 2: Apply softmax to each row with parallelization
    with nogil, parallel():
        for b in prange(batch_size):
            for i in range(seq_len):
                # Find max for numerical stability
                max_val = QKt[b, i, 0]
                for j in range(1, seq_len):
                    if QKt[b, i, j] > max_val:
                        max_val = QKt[b, i, j]
                
                # Compute exp and sum
                sum_exp = 0.0
                for j in range(seq_len):
                    attention_weights[b, i, j] = exp(QKt[b, i, j] - max_val)
                    sum_exp += attention_weights[b, i, j]
                
                # Normalize
                for j in range(seq_len):
                    attention_weights[b, i, j] /= sum_exp
    
    # Step 3: Compute the final output (attention_weights @ V) in blocks using AVX
    for b in range(batch_size):
        for i_block in range(0, seq_len, block_size):
            i_end = min(i_block + block_size, seq_len)
            for d_block in range(0, d_model, block_size):
                d_end = min(d_block + block_size, d_model)
                
                # Use our optimized AVX function for this block
                avx_attention_output_block(
                    attention_weights_ptr, V_ptr, output_ptr,
                    b, i_block, i_end, d_block, d_end,
                    seq_len, d_model
                )
    
    return output 