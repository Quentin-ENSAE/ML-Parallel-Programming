import numpy as np
from .attention import cython_block_attention, cython_optimized_attention, cython_avx_attention

def cython_attention(Q, K, V):
    """
    Wrapper function for the Cython implementation of attention
    
    Parameters:
    -----------
    Q : ndarray of shape (batch_size, seq_len, d_model)
        Query matrix
    K : ndarray of shape (batch_size, seq_len, d_model)
        Key matrix
    V : ndarray of shape (batch_size, seq_len, d_model)
        Value matrix
        
    Returns:
    --------
    output : ndarray of shape (batch_size, seq_len, d_model)
        Result of the attention operation
    """
    # Determine optimal block size based on sequence length
    seq_len = Q.shape[1]
    if seq_len <= 32:
        block_size = 8
    elif seq_len <= 64:
        block_size = 16
    else:
        block_size = 32
        
    return cython_block_attention(Q, K, V, block_size=block_size)

def optimized_attention(Q, K, V,block_size=16):
    """
    Wrapper function for the optimized Cython implementation of attention
    
    Parameters:
    -----------
    Q : ndarray of shape (batch_size, seq_len, d_model)
        Query matrix
    K : ndarray of shape (batch_size, seq_len, d_model)
        Key matrix
    V : ndarray of shape (batch_size, seq_len, d_model)
        Value matrix
        
    Returns:
    --------
    output : ndarray of shape (batch_size, seq_len, d_model)
        Result of the attention operation
    """
    return cython_optimized_attention(Q, K, V,block_size=block_size) 

def avx_attention(Q, K, V,block_size=16):
    """
    Wrapper function for the AVX implementation of attention
    """
    return cython_avx_attention(Q, K, V,block_size=block_size) 
