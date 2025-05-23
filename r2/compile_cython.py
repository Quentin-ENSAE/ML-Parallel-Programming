"""
Script to compile the Cython extension for the attention implementation
"""
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

def compile_cython_extension():
    console = Console()
    console.print(Panel.fit("[bold blue]Compiling Cython Extension for Attention Implementation", 
                           subtitle="This will compile the parallel block-based attention function"))
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cython_dir = os.path.join(current_dir, "cython")
    
    # Change to the cython directory
    os.chdir(cython_dir)
    
    try:
        # Run the setup.py script
        console.print("[bold yellow]Running: python setup.py build_ext --inplace")
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
        console.print("[bold green]Successfully compiled Cython extension!")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error compiling Cython extension: {e}")
        console.print("[yellow]Make sure you have the necessary build tools and libraries installed.")
        console.print("[yellow]You may need to install: gcc, python-dev, numpy, and Cython.")
        return False
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}")
        return False
    finally:
        # Change back to the original directory
        os.chdir(current_dir)

if __name__ == "__main__":
    success = compile_cython_extension()
    sys.exit(0 if success else 1) 