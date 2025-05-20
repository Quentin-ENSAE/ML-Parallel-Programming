import numpy as np
from scipy.special import softmax
import time
import matplotlib.pyplot as plt

def numpy_attention(Q, K, V):
    QKt = np.matmul(Q, np.transpose(K, (0, 2, 1)))
    d_k = K.shape[-1]
    QKt_scaled = QKt / np.sqrt(d_k)
    attention_weights = softmax(QKt_scaled, axis=-1)
    attention_output = np.matmul(attention_weights, V) 
    return attention_output

def naive_attention(Q, K, V):
    batch_size, seq_len, d_model = Q.shape
    d_k = K.shape[-1]
    
    # Initialize output array
    output = np.zeros((batch_size, seq_len, d_model))
    
    # For each item in the batch
    for b in range(batch_size):
        # For each query position
        for i in range(seq_len):
            # Calculate attention scores for this position
            scores = np.zeros(seq_len)
            for j in range(seq_len):
                # Manual dot product between query and key
                dot_product = 0
                for d in range(d_model):
                    dot_product += Q[b, i, d] * K[b, j, d]
                scores[j] = dot_product / np.sqrt(d_k)
            
            # Apply softmax manually
            exp_scores = np.exp(scores - np.max(scores))  # For numerical stability
            weights = exp_scores / np.sum(exp_scores)
            
            # Weighted sum of values
            for d in range(d_model):
                for j in range(seq_len):
                    output[b, i, d] += weights[j] * V[b, j, d]
    
    return output

def benchmark_attention(batch_sizes, seq_lengths, d_models, repeats=3):
    results = []
    
    print("\nBenchmarking attention with different dimensions:")
    print("implementation | batch_size | seq_length | d_model | time (ms)")
    print("-" * 70)
    
    for b in batch_sizes:
        for l in seq_lengths:
            for d in d_models:
                # Create random input matrices
                Q = np.random.randn(b, l, d)
                K = np.random.randn(b, l, d) 
                V = np.random.randn(b, l, d)
                
                # Benchmark numpy implementation
                numpy_times = []
                for _ in range(repeats):
                    start = time.time()
                    _ = numpy_attention(Q, K, V)
                    end = time.time()
                    numpy_times.append((end - start) * 1000)
                
                numpy_time = np.mean(numpy_times)
                print(f"{'numpy':>13} | {b:>10} | {l:>10} | {d:>7} | {numpy_time:>8.2f}")
                results.append(('numpy', b, l, d, numpy_time))
                
                # Skip naive implementation for larger dimensions as it would be too slow
                if b <= 4 and l <= 64 and d <= 128:
                    naive_times = []
                    for _ in range(repeats):
                        start = time.time()
                        _ = naive_attention(Q, K, V)
                        end = time.time()
                        naive_times.append((end - start) * 1000)
                    
                    naive_time = np.mean(naive_times)
                    print(f"{'naive':>13} | {b:>10} | {l:>10} | {d:>7} | {naive_time:>8.2f}")
                    results.append(('naive', b, l, d, naive_time))
    
    return results

def plot_results(results):
    # Organize results
    numpy_results = [(b, l, d, t) for impl, b, l, d, t in results if impl == 'numpy']
    naive_results = [(b, l, d, t) for impl, b, l, d, t in results if impl == 'naive']
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Impact of sequence length on numpy implementation
    plt.subplot(2, 2, 1)
    seq_lengths = sorted(list(set([l for _, l, _, _ in numpy_results])))
    
    for d in sorted(list(set([d for _, _, d, _ in numpy_results])))[:2]:  # Limit to first 2 d_models for clarity
        for b in sorted(list(set([b for b, _, _, _ in numpy_results])))[:2]:  # Limit to first 2 batch sizes for clarity
            times = []
            seq_lens_available = []
            
            for seq_len in seq_lengths:
                matching_results = [(b_, l_, d_, t_) for b_, l_, d_, t_ in numpy_results if b_ == b and l_ == seq_len and d_ == d]
                if matching_results:
                    times.append(matching_results[0][3])
                    seq_lens_available.append(seq_len)
            
            if times:
                plt.plot(seq_lens_available, times, marker='o', label=f'batch={b}, d_model={d}')
    
    plt.title('NumPy Implementation: Impact of Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Impact of d_model on numpy implementation
    plt.subplot(2, 2, 2)
    d_model_sizes = sorted(list(set([d for _, _, d, _ in numpy_results])))
    
    for l in sorted(list(set([l for _, l, _, _ in numpy_results])))[:2]:  # Limit to first 2 sequence lengths for clarity
        for b in sorted(list(set([b for b, _, _, _ in numpy_results])))[:2]:  # Limit to first 2 batch sizes for clarity
            times = []
            d_models_available = []
            
            for d_model in d_model_sizes:
                matching_results = [(b_, l_, d_, t_) for b_, l_, d_, t_ in numpy_results if b_ == b and l_ == l and d_ == d_model]
                if matching_results:
                    times.append(matching_results[0][3])
                    d_models_available.append(d_model)
            
            if times:
                plt.plot(d_models_available, times, marker='o', label=f'batch={b}, seq_len={l}')
    
    plt.title('NumPy Implementation: Impact of d_model')
    plt.xlabel('d_model')
    plt.ylabel('Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 3: Comparison between implementations
    if naive_results:
        plt.subplot(2, 2, 3)
        
        # Get common parameters for both implementations
        common_params = []
        for b, l, d, _ in numpy_results:
            for b2, l2, d2, _ in naive_results:
                if b == b2 and l == l2 and d == d2:
                    common_params.append((b, l, d))
        
        common_params = sorted(list(set(common_params)))
        
        if common_params:
            numpy_times = []
            naive_times = []
            param_labels = []
            
            for b, l, d in common_params:
                numpy_time = next((t for b_, l_, d_, t in numpy_results if b_ == b and l_ == l and d_ == d), None)
                naive_time = next((t for b_, l_, d_, t in naive_results if b_ == b and l_ == l and d_ == d), None)
                
                if numpy_time is not None and naive_time is not None:
                    numpy_times.append(numpy_time)
                    naive_times.append(naive_time)
                    param_labels.append(f"b{b}-l{l}-d{d}")
            
            if numpy_times and naive_times:
                x = np.arange(len(param_labels))
                width = 0.35
                
                plt.bar(x - width/2, numpy_times, width, label='NumPy')
                plt.bar(x + width/2, naive_times, width, label='Naive')
                
                plt.xticks(x, param_labels, rotation=45)
                plt.title('NumPy vs Naive Implementation')
                plt.xlabel('Parameters (batch-length-dimension)')
                plt.ylabel('Time (ms)')
                plt.yscale('log')  # Log scale to better see the difference
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Plot 4: Speedup ratios
                plt.subplot(2, 2, 4)
                speedups = [naive_t / numpy_t for numpy_t, naive_t in zip(numpy_times, naive_times)]
                
                plt.bar(param_labels, speedups, color='teal')
                plt.title('Speedup Ratio (Naive / NumPy)')
                plt.xlabel('Parameters (batch-length-dimension)')
                plt.ylabel('Speedup Ratio (log scale)')
                plt.yscale('log')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('attention_benchmark_results.png', dpi=300)
    print("\nBenchmark results plotted and saved to 'attention_benchmark_results.png'")

# Test with different dimensions - more intensive but manageable
# Reduce the parameter space for faster execution
batch_sizes = [1, 2, 4, 8]  
seq_lengths = [16, 32, 64, 128]
d_models = [32, 64, 128, 256]

# Run the benchmark
results = benchmark_attention(batch_sizes, seq_lengths, d_models)
plot_results(results)
