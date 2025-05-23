import numpy as np
import sys
import os

# Add the parent directory to the path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from r2.cython.attention_wrapper import cython_attention, optimized_attention, avx_attention

def numpy_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def numpy_attention(Q, K, V):
    '''
    Numpy implementation of the attention formula
    Q: (batch_size, seq_len, d_model)
    K: (batch_size, seq_len, d_model)
    V: (batch_size, seq_len, d_model)
    '''
    QKt = np.matmul(Q, np.transpose(K, (0, 2, 1)))
    d_k = K.shape[-1]
    QKt_scaled = QKt / np.sqrt(d_k)
    attention_weights = numpy_softmax(QKt_scaled)
    attention_output = np.matmul(attention_weights, V) 
    return attention_output

def naive_attention(Q, K, V):
    '''
    Naive implementation of the attention formula using for loops
    Q: (batch_size, seq_len, d_model)
    K: (batch_size, seq_len, d_model)
    V: (batch_size, seq_len, d_model)
    '''
    batch_size, seq_len, d_model = Q.shape
    d_k = K.shape[-1]
    
    output = np.zeros((batch_size, seq_len, d_model))
    
    for b in range(batch_size):
        for i in range(seq_len):
            scores = np.zeros(seq_len)
            for j in range(seq_len):
                dot_product = 0
                for d in range(d_model):
                    dot_product += Q[b, i, d] * K[b, j, d]
                scores[j] = dot_product / np.sqrt(d_k)
            
            exp_scores = np.exp(scores - np.max(scores))
            weights = exp_scores / np.sum(exp_scores)
            
            for d in range(d_model):
                for j in range(seq_len):
                    output[b, i, d] += weights[j] * V[b, j, d]
    
    return output
