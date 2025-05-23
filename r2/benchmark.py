import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from rich.panel import Panel

from attention_utils import numpy_attention, naive_attention, cython_attention, optimized_attention, avx_attention

def benchmark_attention(functions, batch_sizes, seq_lengths, d_models, repeats=10, save_results=False):
    console = Console()
    console.print(Panel.fit("[bold green]Starting Attention Implementation Benchmark", 
                           subtitle="Comparing different implementations across various parameters"))

    results_df = pd.DataFrame(columns=['function', 'batch_size', 'seq_length', 'd_model', 'time'])
    
    total_iterations = len(batch_sizes) * len(seq_lengths) * len(d_models) * len(functions)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True
    ) as progress:
        main_task = progress.add_task("[bold green]Overall Progress", total=total_iterations)
        batch_task = progress.add_task("[bold cyan]Batch Size", total=len(batch_sizes), visible=False)
        seq_task = progress.add_task("[bold yellow]Sequence Length", total=len(seq_lengths), visible=False)
        dim_task = progress.add_task("[bold magenta]Dimension", total=len(d_models), visible=False)
        func_task = progress.add_task("[bold blue]Function", total=len(functions), visible=False)
        
        for b_idx, b in enumerate(batch_sizes):
            progress.update(batch_task, completed=b_idx, description=f"[bold cyan]Batch Size: {b}")
            progress.update(batch_task, visible=True)
            
            for l_idx, l in enumerate(seq_lengths):
                progress.update(seq_task, completed=l_idx, description=f"[bold yellow]Sequence Length: {l}")
                progress.update(seq_task, visible=True)
                
                for d_idx, d in enumerate(d_models):
                    progress.update(dim_task, completed=d_idx, description=f"[bold magenta]Dimension: {d}")
                    progress.update(dim_task, visible=True)
                    
                    # Generate data once per configuration
                    Q = np.random.randn(b, l, d)
                    K = np.random.randn(b, l, d) 
                    V = np.random.randn(b, l, d)
                    
                    for f_idx, function in enumerate(functions):
                        name = function.__name__
                        progress.update(func_task, completed=f_idx, description=f"[bold blue]Testing: {name}")
                        progress.update(func_task, visible=True)
                        
                        # Create a subtask for the repeats
                        repeat_task = progress.add_task(f"[cyan]Running {repeats} trials", total=repeats, visible=True)
                        
                        times = []
                        for r in range(repeats):
                            start = time.time()
                            _ = function(Q, K, V)
                            end = time.time()
                            times.append((end - start) * 1000)
                            
                            # Update the repeat progress
                            progress.update(repeat_task, completed=r+1)
                            
                        # Remove the repeat task when done
                        progress.remove_task(repeat_task)
                        
                        # Log the result
                        avg_time = np.mean(times)
                        results_df = pd.concat([results_df, pd.DataFrame({
                            'function': [name], 
                            'batch_size': [b], 
                            'seq_length': [l], 
                            'd_model': [d], 
                            'time': [avg_time]
                        })], ignore_index=True)
                        
                        # Update the main progress bar
                        progress.update(main_task, advance=1)
                        
                        # Print a summary of this configuration
                        console.print(f"[dim]{name}: b={b}, l={l}, d={d} → {avg_time:.2f} ms")

    if save_results:
        console.print(f"[bold green]Saving results to attention_benchmark_results.csv")
        results_df.to_csv('attention_benchmark_results.csv', index=False)
    
    console.print(Panel.fit("[bold green]Benchmark Complete!", 
                           subtitle=f"Tested {len(functions)} functions across {total_iterations} configurations"))

    return results_df

def plot_results(results_df):
    """
    Create a dashboard-style visualization of benchmark results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing benchmark results with columns:
        'function', 'batch_size', 'seq_length', 'd_model', 'time'
    """
    console = Console()
    console.print(Panel.fit("[bold blue]Generating Visualization Dashboard", 
                           subtitle="Creating comparative plots of benchmark results"))
    
    plt.figure(figsize=(20, 15))
    plt.style.use('ggplot')
    
    # Set up the subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Attention Implementation Benchmark Results', fontsize=20)
    
    # 1. Plot time vs batch_size for different functions
    pivot_batch = results_df.pivot_table(
        index='batch_size', 
        columns='function', 
        values='time', 
        aggfunc='mean'
    )
    pivot_batch.plot(
        kind='bar', 
        ax=axes[0, 0], 
        title='Average Execution Time by Batch Size'
    )
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].legend(title='Implementation')
    
    # 2. Plot time vs sequence_length for different functions
    pivot_seq = results_df.pivot_table(
        index='seq_length', 
        columns='function', 
        values='time', 
        aggfunc='mean'
    )
    pivot_seq.plot(
        kind='line', 
        marker='o', 
        ax=axes[0, 1], 
        title='Average Execution Time by Sequence Length'
    )
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].legend(title='Implementation')
    
    # 3. Plot time vs d_model for different functions
    pivot_d = results_df.pivot_table(
        index='d_model', 
        columns='function', 
        values='time', 
        aggfunc='mean'
    )
    pivot_d.plot(
        kind='line', 
        marker='o', 
        ax=axes[1, 0], 
        title='Average Execution Time by Model Dimension'
    )
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].set_xlabel('Model Dimension (d_model)')
    axes[1, 0].legend(title='Implementation')
    
    # 4. Heatmap of speedup ratio
    # Get the unique functions
    functions = results_df['function'].unique()
    if len(functions) >= 2:
        # Create a pivot table for the speedup ratio
        speedup_df = results_df.pivot_table(
            index='seq_length', 
            columns='batch_size', 
            values='time', 
            aggfunc='mean'
        )
        
        # Plot the heatmap
        sns.heatmap(
            speedup_df, 
            annot=True, 
            fmt='.1f', 
            cmap='YlGnBu', 
            ax=axes[1, 1],
            cbar_kws={'label': 'Time (ms)'}
        )
        axes[1, 1].set_title('Execution Time Heatmap (Sequence Length vs Batch Size)')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Sequence Length')
    else:
        # If we only have one function, show a different plot
        sns.boxplot(
            x='d_model', 
            y='time', 
            hue='seq_length', 
            data=results_df, 
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Distribution of Execution Times')
        axes[1, 1].set_xlabel('Model Dimension (d_model)')
        axes[1, 1].set_ylabel('Time (ms)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    console.print("[bold green]Saving visualization as attention_benchmark_results.png")
    plt.savefig('attention_benchmark_results.png', dpi=300)
    
    console.print(Panel.fit("[bold green]Visualization Complete!", 
                           subtitle="Dashboard saved as attention_benchmark_results.png"))
    
    plt.show()

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
seq_lengths = [16, 32, 64, 128 ]
d_models = [32, 64, 128]

functions = [numpy_attention, cython_attention]
# Create named functions for each block size
for bs in [8, 16, 32, 64, 128, 256, 512]:
    # Use a function factory to properly capture the block_size
    def create_named_function(block_size):
        def optimized_wrapper(Q, K, V):
            return optimized_attention(Q, K, V, block_size=block_size)
        optimized_wrapper.__name__ = f"optimized_bs{block_size}"
        return optimized_wrapper
    
    functions.append(create_named_function(bs))

    def create_named_function(block_size):
        def avx_wrapper(Q, K, V):
            return avx_attention(Q, K, V, block_size=block_size)
        avx_wrapper.__name__ = f"avx_bs{block_size}"
        return avx_wrapper
    
    functions.append(create_named_function(bs))

results_df = benchmark_attention(functions, batch_sizes, seq_lengths, d_models, repeats=10)
plot_results(results_df)
