# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
from cython.parallel import prange
from scipy.special import softmax

# Define dtypes
ctypedef np.float64_t DTYPE_t

def cython_attention(np.ndarray[DTYPE_t, ndim=3] Q, 
                     np.ndarray[DTYPE_t, ndim=3] K,
                     np.ndarray[DTYPE_t, ndim=3] V):
    """
    Cython implementation of the attention mechanism
    """
    cdef int batch_size = Q.shape[0]
    cdef int seq_len = Q.shape[1]
    cdef int d_model = Q.shape[2]
    cdef int d_k = K.shape[2]
    
    # Initialize output array
    cdef np.ndarray[DTYPE_t, ndim=3] QKt = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] attention_weights
    cdef np.ndarray[DTYPE_t, ndim=3] output = np.zeros((batch_size, seq_len, d_model), dtype=np.float64)
    
    # Calculate QK^T
    _compute_qkt(Q, K, QKt, batch_size, seq_len, d_model, d_k)
    
    # Apply softmax to get attention weights
    attention_weights = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    for b in range(batch_size):
        for i in range(seq_len):
            attention_weights[b, i] = softmax(QKt[b, i], axis=-1)
    
    # Calculate output
    _compute_output(attention_weights, V, output, batch_size, seq_len, d_model)
    
    return output

def cython_attention_parallel(np.ndarray[DTYPE_t, ndim=3] Q, 
                             np.ndarray[DTYPE_t, ndim=3] K,
                             np.ndarray[DTYPE_t, ndim=3] V, 
                             int num_threads=4):
    """
    Parallel Cython implementation of the attention mechanism
    """
    cdef int batch_size = Q.shape[0]
    cdef int seq_len = Q.shape[1]
    cdef int d_model = Q.shape[2]
    cdef int d_k = K.shape[2]
    
    # Initialize output array
    cdef np.ndarray[DTYPE_t, ndim=3] QKt = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] attention_weights
    cdef np.ndarray[DTYPE_t, ndim=3] output = np.zeros((batch_size, seq_len, d_model), dtype=np.float64)
    
    # Calculate QK^T
    _compute_dot_products_parallel(Q, K, QKt, batch_size, seq_len, d_model, d_k, num_threads)
    
    # Apply softmax to get attention weights
    attention_weights = np.zeros((batch_size, seq_len, seq_len), dtype=np.float64)
    for b in range(batch_size):
        for i in range(seq_len):
            attention_weights[b, i] = softmax(QKt[b, i], axis=-1)
    
    # Calculate output
    _compute_attention_output_parallel(attention_weights, V, output, batch_size, seq_len, d_model, num_threads)
    
    return output

# Helper functions
cdef void _compute_qkt(DTYPE_t[:, :, :] Q, 
                      DTYPE_t[:, :, :] K, 
                      DTYPE_t[:, :, :] QKt,
                      int batch_size, 
                      int seq_len, 
                      int d_model, 
                      int d_k) nogil:
    cdef int b, i, j, d
    cdef double dot_product, scale
    
    scale = 1.0 / sqrt(d_k)
    
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                dot_product = 0.0
                for d in range(d_model):
                    dot_product += Q[b, i, d] * K[b, j, d]
                QKt[b, i, j] = dot_product * scale

cdef void _compute_dot_products_parallel(DTYPE_t[:, :, :] Q, 
                                       DTYPE_t[:, :, :] K, 
                                       DTYPE_t[:, :, :] QKt,
                                       int batch_size, 
                                       int seq_len, 
                                       int d_model, 
                                       int d_k,
                                       int num_threads) nogil:
    cdef int b, i, j, d
    cdef double scale = 1.0 / sqrt(d_k)
    
    # Parallelize over batches
    for b in prange(batch_size, nogil=True, num_threads=num_threads):
        for i in range(seq_len):
            for j in range(seq_len):
                QKt[b, i, j] = 0.0
                for d in range(d_model):
                    QKt[b, i, j] += Q[b, i, d] * K[b, j, d]
                QKt[b, i, j] *= scale

cdef void _compute_output(DTYPE_t[:, :, :] attention_weights, 
                         DTYPE_t[:, :, :] V, 
                         DTYPE_t[:, :, :] output,
                         int batch_size, 
                         int seq_len, 
                         int d_model) nogil:
    cdef int b, i, j, d
    
    for b in range(batch_size):
        for i in range(seq_len):
            for d in range(d_model):
                for j in range(seq_len):
                    output[b, i, d] += attention_weights[b, i, j] * V[b, j, d]

cdef void _compute_attention_output_parallel(DTYPE_t[:, :, :] attention_weights, 
                                           DTYPE_t[:, :, :] V, 
                                           DTYPE_t[:, :, :] output,
                                           int batch_size, 
                                           int seq_len, 
                                           int d_model,
                                           int num_threads) nogil:
    cdef int b, i, d, j
    
    # Parallelize over batches
    for b in prange(batch_size, nogil=True, num_threads=num_threads):
        for i in range(seq_len):
            for d in range(d_model):
                output[b, i, d] = 0.0
                for j in range(seq_len):
                    output[b, i, d] += attention_weights[b, i, j] * V[b, j, d] 