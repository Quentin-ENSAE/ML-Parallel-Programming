import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from benchmark_olk import numpy_attention
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from attention_parallel import cython_attention, cython_attention_parallel
import os
import multiprocessing

# Get the number of CPU cores available
MAX_CORES = multiprocessing.cpu_count()

def verify_implementations(Q, K, V):
    """
    Verify that all implementations produce the same results.
    """
    numpy_output = numpy_attention(Q, K, V)
    cython_output = cython_attention(Q, K, V)
    cython_parallel_output = cython_attention_parallel(Q, K, V, num_threads=2)
    
    # Check if outputs are close enough (allowing for small numerical differences)
    numpy_cython_close = np.allclose(numpy_output, cython_output, rtol=1e-5, atol=1e-5)
    numpy_parallel_close = np.allclose(numpy_output, cython_parallel_output, rtol=1e-5, atol=1e-5)
    
    if not numpy_cython_close:
        print("Warning: Cython implementation output differs from NumPy implementation")
    
    if not numpy_parallel_close:
        print("Warning: Parallel Cython implementation output differs from NumPy implementation")
    
    return numpy_cython_close and numpy_parallel_close

def run_single_benchmark(implementation, Q, K, V, num_threads=None, repeats=3):
    """
    Run a single benchmark for the given implementation and parameters.
    Returns the average time in milliseconds.
    """
    times = []
    
    for _ in range(repeats):
        if implementation == 'numpy':
            start = time.time()
            _ = numpy_attention(Q, K, V)
            end = time.time()
        elif implementation == 'cython':
            start = time.time()
            _ = cython_attention(Q, K, V)
            end = time.time()
        elif implementation == 'cython_parallel':
            start = time.time()
            _ = cython_attention_parallel(Q, K, V, num_threads=num_threads)
            end = time.time()
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        times.append((end - start) * 1000)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    ci = stats.t.interval(0.95, len(times)-1, loc=mean_time, scale=std_time/np.sqrt(len(times)))
    
    return {
        'mean': mean_time,
        'std': std_time,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'times': times
    }

def objective(batch_size, seq_length, d_model, num_threads):
    """
    Objective function for Bayesian optimization.
    Returns the execution time (to be minimized).
    """
    # Create random input matrices
    Q = np.random.randn(batch_size, seq_length, d_model)
    K = np.random.randn(batch_size, seq_length, d_model)
    V = np.random.randn(batch_size, seq_length, d_model)
    
    # Run benchmark for parallel implementation
    result = run_single_benchmark('cython_parallel', Q, K, V, num_threads=num_threads)
    
    return result['mean']

# Define the search space
dimensions = [
    Integer(1, MAX_CORES, name='num_threads'),
    Integer(1, 16, name='batch_size'),
    Integer(16, 256, name='seq_length'),
    Integer(32, 512, name='d_model')
]

@use_named_args(dimensions)
def objective_wrapper(num_threads, batch_size, seq_length, d_model):
    """
    Wrapper for the objective function that accepts named arguments.
    """
    return objective(batch_size, seq_length, d_model, num_threads)

def adaptive_benchmark(n_calls=30, random_state=42):
    """
    Run the adaptive benchmark using Bayesian optimization to find the best configuration.
    """
    print(f"Starting adaptive benchmark with {n_calls} evaluations")
    print("Searching for optimal configuration...")
    
    # Run the optimization
    result = gp_minimize(
        objective_wrapper,
        dimensions,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True,
        n_random_starts=10  # Start with 10 random evaluations before using the surrogate model
    )
    
    # Extract the best parameters
    best_params = {
        'num_threads': result.x[0],
        'batch_size': result.x[1],
        'seq_length': result.x[2],
        'd_model': result.x[3]
    }
    
    print("\nBest configuration found:")
    print(f"Number of threads: {best_params['num_threads']}")
    print(f"Batch size: {best_params['batch_size']}")
    print(f"Sequence length: {best_params['seq_length']}")
    print(f"Model dimension: {best_params['d_model']}")
    print(f"Execution time: {result.fun:.2f} ms")
    
    # Save the optimization results
    with open('optimization_results.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    return result, best_params

def validate_best_configuration(best_params, grid_validation=True):
    """
    Validate the best configuration found with Bayesian optimization.
    Optionally compare against a grid search on a subset of the parameter space.
    """
    # Create input matrices with the best parameters
    Q = np.random.randn(best_params['batch_size'], best_params['seq_length'], best_params['d_model'])
    K = np.random.randn(best_params['batch_size'], best_params['seq_length'], best_params['d_model'])
    V = np.random.randn(best_params['batch_size'], best_params['seq_length'], best_params['d_model'])
    
    # Run benchmarks for all implementations
    numpy_result = run_single_benchmark('numpy', Q, K, V)
    cython_result = run_single_benchmark('cython', Q, K, V)
    cython_parallel_result = run_single_benchmark(
        'cython_parallel', Q, K, V, num_threads=best_params['num_threads']
    )
    
    print("\nValidation of the best configuration:")
    print(f"NumPy implementation: {numpy_result['mean']:.2f} ms")
    print(f"Cython implementation: {cython_result['mean']:.2f} ms")
    print(f"Parallel Cython implementation ({best_params['num_threads']} threads): {cython_parallel_result['mean']:.2f} ms")
    
    speedup_over_numpy = numpy_result['mean'] / cython_parallel_result['mean']
    speedup_over_cython = cython_result['mean'] / cython_parallel_result['mean']
    
    print(f"Speedup over NumPy: {speedup_over_numpy:.2f}x")
    print(f"Speedup over Cython: {speedup_over_cython:.2f}x")
    
    # If requested, run a small grid search around the best configuration
    if grid_validation:
        print("\nValidating with grid search around best configuration...")
        
        # Define ranges around the best values
        thread_range = list(range(max(1, best_params['num_threads'] - 2), 
                                  min(MAX_CORES, best_params['num_threads'] + 3)))
        
        grid_results = []
        
        for threads in thread_range:
            result = run_single_benchmark('cython_parallel', Q, K, V, num_threads=threads)
            grid_results.append({
                'num_threads': threads,
                'time': result['mean']
            })
            print(f"Number of threads: {threads}, Execution time: {result['mean']:.2f} ms")
        
        # Find the best from the grid search
        best_grid = min(grid_results, key=lambda x: x['time'])
        print(f"\nBest from grid search: {best_grid['num_threads']} threads with {best_grid['time']:.2f} ms")
        
        return {
            'bayesian_optimal': cython_parallel_result,
            'grid_optimal': best_grid,
            'numpy': numpy_result,
            'cython': cython_result,
            'speedup_over_numpy': speedup_over_numpy,
            'speedup_over_cython': speedup_over_cython
        }
    
    return {
        'bayesian_optimal': cython_parallel_result,
        'numpy': numpy_result,
        'cython': cython_result,
        'speedup_over_numpy': speedup_over_numpy,
        'speedup_over_cython': speedup_over_cython
    }

def plot_optimization_results(result):
    """
    Plot the results of the Bayesian optimization process.
    """
    plt.figure(figsize=(15, 14))
    
    # Plot 1: Convergence plot
    plt.subplot(3, 2, 1)
    plot_convergence(result)
    plt.title('Convergence of Bayesian Optimization')
    
    # Plot 2: Parameter importance - handle different GP model versions
    plt.subplot(3, 2, 2)
    param_names = ['num_threads', 'batch_size', 'seq_length', 'd_model']
    
    # Try to access theta_ attribute, but handle cases where it might not exist
    try:
        if hasattr(result.models[-1], 'theta_'):
            param_importance = np.abs(result.models[-1].theta_)
        elif hasattr(result.models[-1], 'kernel_'):
            # For newer scikit-optimize versions
            param_importance = np.abs(result.models[-1].kernel_.get_params().get('k2__k1__k1__length_scale', [1, 1, 1, 1]))
            if not isinstance(param_importance, np.ndarray):
                param_importance = np.array([1, 1, 1, 1])  # Fallback if we can't get the values
        else:
            # Fallback if we can't access the importance values
            param_importance = np.array([1, 1, 1, 1])
        
        # Ensure correct length
        if len(param_importance) != 4:
            param_importance = np.ones(4)
            
        plt.bar(param_names, param_importance)
        plt.title('Parameter Importance (estimated)')
        plt.ylabel('Importance (length-scale)')
        plt.xticks(rotation=45)
    except Exception as e:
        plt.text(0.5, 0.5, f"Could not plot parameter importance: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Parameter Importance (not available)')
    
    # Plot 3: Scatter plot of evaluations - threads vs time
    plt.subplot(3, 2, 3)
    threads = [x[0] for x in result.x_iters]
    times = result.func_vals
    plt.scatter(threads, times)
    plt.plot(threads, times, 'r--', alpha=0.3)
    plt.title('Number of Threads vs. Execution Time')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (ms)')
    
    # Plot 4: Distribution of execution times
    plt.subplot(3, 2, 4)
    plt.hist(result.func_vals, bins=10)
    plt.axvline(result.fun, color='r', linestyle='--')
    plt.title('Distribution of Execution Times')
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')
    
    # Plot 5: Scatter plot of batch_size vs seq_length colored by execution time
    plt.subplot(3, 2, 5)
    batch_sizes = [x[1] for x in result.x_iters]
    seq_lengths = [x[2] for x in result.x_iters]
    plt.scatter(batch_sizes, seq_lengths, c=times, cmap='viridis')
    plt.colorbar(label='Execution Time (ms)')
    plt.title('Batch Size vs. Sequence Length')
    plt.xlabel('Batch Size')
    plt.ylabel('Sequence Length')
    
    # Plot 6: Scatter plot of seq_length vs d_model colored by execution time
    plt.subplot(3, 2, 6)
    d_models = [x[3] for x in result.x_iters]
    plt.scatter(seq_lengths, d_models, c=times, cmap='viridis')
    plt.colorbar(label='Execution Time (ms)')
    plt.title('Sequence Length vs. Model Dimension')
    plt.xlabel('Sequence Length')
    plt.ylabel('Model Dimension')
    
    plt.tight_layout()
    plt.savefig('adaptive_benchmark_results.png', dpi=300)
    print("\nOptimization results plotted and saved to 'adaptive_benchmark_results.png'")
    
    # Create an additional plot for dimension analysis
    plt.figure(figsize=(15, 10))
    
    # Plot matrix dimensions impact
    # 1. Effect of sequence length on time
    plt.subplot(2, 2, 1)
    for b in sorted(set(batch_sizes))[:3]:  # Top 3 batch sizes
        batch_indices = [i for i, x in enumerate(batch_sizes) if x == b]
        if batch_indices:
            plt.scatter(
                [seq_lengths[i] for i in batch_indices],
                [times[i] for i in batch_indices],
                label=f'batch_size={b}'
            )
    plt.title('Effect of Sequence Length on Execution Time')
    plt.xlabel('Sequence Length')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    
    # 2. Effect of model dimension on time
    plt.subplot(2, 2, 2)
    for s in sorted(set(seq_lengths))[:3]:  # Top 3 sequence lengths
        seq_indices = [i for i, x in enumerate(seq_lengths) if x == s]
        if seq_indices:
            plt.scatter(
                [d_models[i] for i in seq_indices],
                [times[i] for i in seq_indices],
                label=f'seq_length={s}'
            )
    plt.title('Effect of Model Dimension on Execution Time')
    plt.xlabel('Model Dimension')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    
    # 3. Effect of batch size on time
    plt.subplot(2, 2, 3)
    for d in sorted(set(d_models))[:3]:  # Top 3 dimensions
        dim_indices = [i for i, x in enumerate(d_models) if x == d]
        if dim_indices:
            plt.scatter(
                [batch_sizes[i] for i in dim_indices],
                [times[i] for i in dim_indices],
                label=f'd_model={d}'
            )
    plt.title('Effect of Batch Size on Execution Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    
    # 4. 3D scatter plot if we can for those who have matplotlib 3D
    try:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(2, 2, 4, projection='3d')
        scatter = ax.scatter(batch_sizes, seq_lengths, d_models, c=times, cmap='viridis')
        plt.colorbar(scatter, label='Execution Time (ms)')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Sequence Length')
        ax.set_zlabel('Model Dimension')
        ax.set_title('3D View of Parameter Space')
    except Exception:
        plt.subplot(2, 2, 4)
        # Create a pseudo-3D visualization with size encoding d_model
        sizes = [max(10, min(1000, d*0.5)) for d in d_models]
        scatter = plt.scatter(batch_sizes, seq_lengths, s=sizes, c=times, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Execution Time (ms)')
        plt.title('Parameter Space (size = model dimension)')
        plt.xlabel('Batch Size')
        plt.ylabel('Sequence Length')
    
    plt.tight_layout()
    plt.savefig('dimension_analysis.png', dpi=300)
    print("Additional dimension analysis plotted and saved to 'dimension_analysis.png'")

def compare_implementations(best_params):
    """
    Compare different implementations across a range of dimensions.
    """
    print("\nRunning detailed implementation comparison...")
    
    # Define ranges for the parameters
    thread_range = [1, 2, 4, 8] if MAX_CORES >= 8 else list(range(1, MAX_CORES + 1))
    batch_range = [1, 2, 4]
    seq_len_range = [16, 32, 64]
    d_model_range = [64, 128, 256]
    
    # Hold all results
    comparison_results = []
    
    # First, test the best configuration found across all implementations and thread counts
    best_Q = np.random.randn(best_params['batch_size'], best_params['seq_length'], best_params['d_model'])
    best_K = np.random.randn(best_params['batch_size'], best_params['seq_length'], best_params['d_model'])
    best_V = np.random.randn(best_params['batch_size'], best_params['seq_length'], best_params['d_model'])
    
    print("\nComparing implementations with the best configuration:")
    print(f"Batch size: {best_params['batch_size']}, Sequence length: {best_params['seq_length']}, Model dimension: {best_params['d_model']}")
    
    # Test NumPy
    numpy_result = run_single_benchmark('numpy', best_Q, best_K, best_V)
    comparison_results.append({
        'implementation': 'numpy',
        'threads': 'N/A',
        'batch_size': best_params['batch_size'],
        'seq_length': best_params['seq_length'],
        'd_model': best_params['d_model'],
        'time': numpy_result['mean']
    })
    print(f"NumPy implementation: {numpy_result['mean']:.2f} ms")
    
    # Test regular Cython
    cython_result = run_single_benchmark('cython', best_Q, best_K, best_V)
    comparison_results.append({
        'implementation': 'cython',
        'threads': 'N/A',
        'batch_size': best_params['batch_size'],
        'seq_length': best_params['seq_length'],
        'd_model': best_params['d_model'],
        'time': cython_result['mean']
    })
    print(f"Cython implementation: {cython_result['mean']:.2f} ms")
    
    # Test parallel Cython with different thread counts
    for threads in thread_range:
        parallel_result = run_single_benchmark('cython_parallel', best_Q, best_K, best_V, num_threads=threads)
        comparison_results.append({
            'implementation': 'cython_parallel',
            'threads': threads,
            'batch_size': best_params['batch_size'],
            'seq_length': best_params['seq_length'],
            'd_model': best_params['d_model'],
            'time': parallel_result['mean']
        })
        print(f"Parallel Cython ({threads} threads): {parallel_result['mean']:.2f} ms")
    
    # Now test a smaller subset of combinations to see dimension impact
    print("\nTesting dimension impact on different implementations...")
    
    # Sample a subset of combinations to avoid explosion
    sample_size = min(20, len(batch_range) * len(seq_len_range) * len(d_model_range))
    combinations = []
    for b in batch_range:
        for s in seq_len_range:
            for d in d_model_range:
                combinations.append((b, s, d))
    
    # Randomly sample if there are too many combinations
    if len(combinations) > sample_size:
        sample_indices = np.random.choice(len(combinations), size=sample_size, replace=False)
        combinations = [combinations[i] for i in sample_indices]
    
    for b, s, d in combinations:
        Q = np.random.randn(b, s, d)
        K = np.random.randn(b, s, d)
        V = np.random.randn(b, s, d)
        
        # Test NumPy
        numpy_result = run_single_benchmark('numpy', Q, K, V)
        comparison_results.append({
            'implementation': 'numpy',
            'threads': 'N/A',
            'batch_size': b,
            'seq_length': s,
            'd_model': d,
            'time': numpy_result['mean']
        })
        
        # Test regular Cython
        cython_result = run_single_benchmark('cython', Q, K, V)
        comparison_results.append({
            'implementation': 'cython',
            'threads': 'N/A',
            'batch_size': b,
            'seq_length': s,
            'd_model': d,
            'time': cython_result['mean']
        })
        
        # Test parallel Cython with best thread count
        best_thread = best_params['num_threads']
        parallel_result = run_single_benchmark('cython_parallel', Q, K, V, num_threads=best_thread)
        comparison_results.append({
            'implementation': 'cython_parallel',
            'threads': best_thread,
            'batch_size': b,
            'seq_length': s,
            'd_model': d,
            'time': parallel_result['mean']
        })
        
        print(f"Batch={b}, Seq={s}, Dim={d}: NumPy={numpy_result['mean']:.2f}ms, " +
              f"Cython={cython_result['mean']:.2f}ms, " +
              f"Parallel({best_thread} threads)={parallel_result['mean']:.2f}ms")
    
    # Plot the comparison results
    plt.figure(figsize=(15, 10))
    
    # Group by implementation and collect data
    numpy_results = [r for r in comparison_results if r['implementation'] == 'numpy']
    cython_results = [r for r in comparison_results if r['implementation'] == 'cython']
    parallel_results = [r for r in comparison_results if r['implementation'] == 'cython_parallel' 
                        and r['threads'] == best_params['num_threads']]
    
    # Plot 1: Sequence length impact across implementations
    plt.subplot(2, 2, 1)
    seq_lengths = sorted(list(set([r['seq_length'] for r in comparison_results])))
    
    for impl, results, marker, label in [
        ('numpy', numpy_results, 'o', 'NumPy'),
        ('cython', cython_results, 's', 'Cython'),
        ('cython_parallel', parallel_results, '^', f'Parallel ({best_params["num_threads"]} threads)')
    ]:
        for s in seq_lengths:
            seq_results = [r for r in results if r['seq_length'] == s]
            if seq_results:
                x = [r['d_model'] for r in seq_results]
                y = [r['time'] for r in seq_results]
                plt.scatter(x, y, marker=marker, label=f'{label}, seq={s}')
    
    plt.title('Effect of Model Dimension by Implementation')
    plt.xlabel('Model Dimension')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    
    # Plot 2: Batch size impact across implementations
    plt.subplot(2, 2, 2)
    batch_sizes = sorted(list(set([r['batch_size'] for r in comparison_results])))
    
    for impl, results, marker, label in [
        ('numpy', numpy_results, 'o', 'NumPy'),
        ('cython', cython_results, 's', 'Cython'),
        ('cython_parallel', parallel_results, '^', f'Parallel ({best_params["num_threads"]} threads)')
    ]:
        for b in batch_sizes:
            batch_results = [r for r in results if r['batch_size'] == b]
            if batch_results:
                x = [r['seq_length'] for r in batch_results]
                y = [r['time'] for r in batch_results]
                plt.scatter(x, y, marker=marker, label=f'{label}, batch={b}')
    
    plt.title('Effect of Sequence Length by Implementation')
    plt.xlabel('Sequence Length')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    
    # Plot 3: Thread count impact for parallel implementation
    plt.subplot(2, 2, 3)
    thread_counts = sorted(list(set([r['threads'] for r in comparison_results if r['implementation'] == 'cython_parallel'])))
    
    for t in thread_counts:
        thread_results = [r for r in comparison_results if r['implementation'] == 'cython_parallel' and r['threads'] == t]
        if thread_results:
            x = [r['d_model'] for r in thread_results]
            y = [r['time'] for r in thread_results]
            plt.scatter(x, y, label=f'{t} threads')
    
    plt.title('Effect of Thread Count on Parallel Implementation')
    plt.xlabel('Model Dimension')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    
    # Plot 4: Implementation comparison for different data sizes
    plt.subplot(2, 2, 4)
    # Create a metric for data size
    for r in comparison_results:
        r['data_size'] = r['batch_size'] * r['seq_length'] * r['d_model']
    
    data_sizes = sorted(list(set([r['data_size'] for r in comparison_results])))
    
    size_groups = {}
    for size in data_sizes:
        numpy_time = np.mean([r['time'] for r in numpy_results if r['data_size'] == size]) if any(r['data_size'] == size for r in numpy_results) else 0
        cython_time = np.mean([r['time'] for r in cython_results if r['data_size'] == size]) if any(r['data_size'] == size for r in cython_results) else 0
        parallel_time = np.mean([r['time'] for r in parallel_results if r['data_size'] == size]) if any(r['data_size'] == size for r in parallel_results) else 0
        
        if numpy_time > 0 and cython_time > 0 and parallel_time > 0:
            size_groups[size] = {
                'numpy': numpy_time,
                'cython': cython_time,
                'parallel': parallel_time
            }
    
    # Take at most 6 data sizes for readability
    if len(size_groups) > 6:
        selected_sizes = sorted(size_groups.keys())[-6:]
        size_groups = {k: size_groups[k] for k in selected_sizes}
    
    # Plot bar chart
    size_labels = [f"{size//1000}K" if size >= 1000 else str(size) for size in size_groups.keys()]
    x = np.arange(len(size_labels))
    width = 0.25
    
    plt.bar(x - width, [size_groups[size]['numpy'] for size in size_groups.keys()], width, label='NumPy')
    plt.bar(x, [size_groups[size]['cython'] for size in size_groups.keys()], width, label='Cython')
    plt.bar(x + width, [size_groups[size]['parallel'] for size in size_groups.keys()], width, label=f'Parallel ({best_params["num_threads"]} threads)')
    
    plt.title('Implementation Performance by Data Size')
    plt.xlabel('Data Size (elements)')
    plt.ylabel('Execution Time (ms)')
    plt.xticks(x, size_labels, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('implementation_comparison.png', dpi=300)
    print("Implementation comparison plotted and saved to 'implementation_comparison.png'")
    
    return comparison_results

def compare_with_baseline_grid(best_params):
    """
    Compare the adaptive approach with a baseline grid search approach.
    """
    print("\nComparing Bayesian optimization vs. Grid Search approach")
    
    # Define a smaller grid to simulate a traditional grid search approach
    batch_sizes = [1, 2, 4, 8]
    seq_lengths = [32, 64, 128]
    d_models = [64, 128, 256]
    thread_counts = [1, 2, 4, 8, 16]
    
    # Count how many evaluations would be needed for a full grid search
    full_grid_evals = len(batch_sizes) * len(seq_lengths) * len(d_models) * len(thread_counts)
    print(f"A full grid search would require {full_grid_evals} evaluations")
    
    # Sample a subset to simulate what a limited grid search might do
    sample_size = min(30, full_grid_evals)  # Same as n_calls in Bayesian opt
    
    print(f"Running a sample of {sample_size} grid points to simulate traditional approach")
    
    grid_results = []
    grid_points = []
    
    # Generate all grid points
    for b in batch_sizes:
        for s in seq_lengths:
            for d in d_models:
                for t in thread_counts:
                    grid_points.append((b, s, d, t))
    
    # Randomly sample from grid points
    sampled_points = np.random.choice(len(grid_points), size=sample_size, replace=False)
    
    for idx in sampled_points:
        b, s, d, t = grid_points[idx]
        time_taken = objective(b, s, d, t)
        grid_results.append({
            'batch_size': b,
            'seq_length': s,
            'd_model': d,
            'num_threads': t,
            'time': time_taken
        })
    
    # Find the best configuration from the grid search
    best_grid = min(grid_results, key=lambda x: x['time'])
    
    print("\nBest configuration from grid search:")
    print(f"Number of threads: {best_grid['num_threads']}")
    print(f"Batch size: {best_grid['batch_size']}")
    print(f"Sequence length: {best_grid['seq_length']}")
    print(f"Model dimension: {best_grid['d_model']}")
    print(f"Execution time: {best_grid['time']:.2f} ms")
    
    # Compare with the Bayesian optimization result
    bayesian_time = objective(
        best_params['batch_size'],
        best_params['seq_length'],
        best_params['d_model'],
        best_params['num_threads']
    )
    
    improvement = (best_grid['time'] - bayesian_time) / best_grid['time'] * 100
    
    print("\nComparison:")
    print(f"Grid Search best time: {best_grid['time']:.2f} ms")
    print(f"Bayesian Optimization best time: {bayesian_time:.2f} ms")
    print(f"Improvement: {improvement:.2f}%")
    
    return {
        'grid_best': best_grid,
        'bayesian_time': bayesian_time,
        'improvement': improvement,
        'grid_results': grid_results
    }

def main():
    # Make sure implementations produce the same results
    print("Verifying implementations...")
    Q = np.random.randn(2, 32, 64)
    K = np.random.randn(2, 32, 64)
    V = np.random.randn(2, 32, 64)
    
    if not verify_implementations(Q, K, V):
        print("Warning: Implementations produce different results. Proceeding anyway.")
    
    # Run the adaptive benchmark
    result, best_params = adaptive_benchmark(n_calls=30)
    
    # Validate the best configuration
    validation_results = validate_best_configuration(best_params)
    
    # Plot the optimization results
    plot_optimization_results(result)
    
    # Compare different implementations
    comparison_results = compare_implementations(best_params)
    
    # Compare with a traditional grid search approach
    grid_comparison = compare_with_baseline_grid(best_params)
    
    print("\nAdaptive benchmark completed successfully.")
    
if __name__ == "__main__":
    main() 