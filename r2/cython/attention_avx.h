#ifndef ATTENTION_AVX_H
#define ATTENTION_AVX_H

#include <immintrin.h>
#include <stdint.h>

// Helper functions for AVX operations in attention mechanism
// These will be used for the dot product operations in the attention mechanism

// AVX for double precision (4 doubles at once)
static inline void avx_dot_product_add_double(const double* a, const double* b, double* result, int length) {
    int i = 0;
    
    // Process 4 doubles at a time using AVX
    for (; i + 3 < length; i += 4) {
        // Load 4 values from a and b
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        
        // Multiply and add
        __m256d prod = _mm256_mul_pd(a_vec, b_vec);
        
        // Horizontal sum of the 4 products
        __m256d sum1 = _mm256_hadd_pd(prod, prod);
        // Extract the lower 128 bits (2 doubles)
        __m128d low = _mm256_extractf128_pd(sum1, 0);
        // Extract the upper 128 bits (2 doubles)
        __m128d high = _mm256_extractf128_pd(sum1, 1);
        // Add the lower and upper parts
        __m128d sum = _mm_add_pd(low, high);
        
        // Add to the result
        *result += _mm_cvtsd_f64(sum);
    }
    
    // Process remaining elements
    for (; i < length; i++) {
        *result += a[i] * b[i];
    }
}

// For matrix-matrix multiplication with AVX (attention QK^T calculation)
static inline void avx_matrix_block_mul_double(
    const double* Q, const double* K, double* QKt,
    int batch_idx, int i_start, int i_end, 
    int j_start, int j_end, int d_model,
    int seq_len) {
    
    // For each position in the output block
    for (int i = i_start; i < i_end; i++) {
        for (int j = j_start; j < j_end; j++) {
            // Calculate the dot product of Q[batch_idx, i, :] and K[batch_idx, j, :]
            double sum = 0.0;
            
            // Q_offset points to Q[batch_idx, i, 0]
            const double* Q_offset = Q + (batch_idx * seq_len * d_model) + (i * d_model);
            // K_offset points to K[batch_idx, j, 0]
            const double* K_offset = K + (batch_idx * seq_len * d_model) + (j * d_model);
            
            // Use AVX for dot product
            avx_dot_product_add_double(Q_offset, K_offset, &sum, d_model);
            
            // Store result in QKt[batch_idx, i, j]
            QKt[(batch_idx * seq_len * seq_len) + (i * seq_len) + j] = sum;
        }
    }
}

// Function for the final attention output calculation (attention_weights @ V)
// This computes output[batch_idx, i_start:i_end, d_start:d_end] using AVX
static inline void avx_attention_output_block(
    const double* attention_weights, const double* V, double* output,
    int batch_idx, int i_start, int i_end,
    int d_start, int d_end, int seq_len, int d_model) {
    
    // For each position in the output block
    for (int i = i_start; i < i_end; i++) {
        for (int d = d_start; d < d_end; d++) {
            // Initialize sum for this output position
            double result = 0.0;
            
            // Special case: try to use AVX2 for groups of 4 sequence elements
            int j = 0;
            for (; j + 3 < seq_len; j += 4) {
                // Load 4 attention weights
                __m256d weights = _mm256_set_pd(
                    attention_weights[(batch_idx * seq_len * seq_len) + (i * seq_len) + j+3],
                    attention_weights[(batch_idx * seq_len * seq_len) + (i * seq_len) + j+2],
                    attention_weights[(batch_idx * seq_len * seq_len) + (i * seq_len) + j+1],
                    attention_weights[(batch_idx * seq_len * seq_len) + (i * seq_len) + j]
                );
                
                // Load 4 value elements from V[:,d]
                __m256d values = _mm256_set_pd(
                    V[(batch_idx * seq_len * d_model) + ((j+3) * d_model) + d],
                    V[(batch_idx * seq_len * d_model) + ((j+2) * d_model) + d],
                    V[(batch_idx * seq_len * d_model) + ((j+1) * d_model) + d],
                    V[(batch_idx * seq_len * d_model) + (j * d_model) + d]
                );
                
                // Multiply and add
                __m256d prod = _mm256_mul_pd(weights, values);
                
                // Horizontal sum
                __m256d sum1 = _mm256_hadd_pd(prod, prod);
                __m128d low = _mm256_extractf128_pd(sum1, 0);
                __m128d high = _mm256_extractf128_pd(sum1, 1);
                __m128d sum = _mm_add_pd(low, high);
                
                // Add to result
                result += _mm_cvtsd_f64(sum);
            }
            
            // Handle remaining elements
            for (; j < seq_len; j++) {
                result += attention_weights[(batch_idx * seq_len * seq_len) + (i * seq_len) + j] *
                          V[(batch_idx * seq_len * d_model) + (j * d_model) + d];
            }
            
            // Store the result
            output[(batch_idx * seq_len * d_model) + (i * d_model) + d] = result;
        }
    }
}

#endif // ATTENTION_AVX_H 