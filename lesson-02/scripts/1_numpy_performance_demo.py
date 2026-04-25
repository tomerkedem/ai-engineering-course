"""
Demonstration of NumPy's performance advantage over regular Python lists.

This script compares the execution time of common array operations
using NumPy arrays versus Python lists.
"""

import numpy as np
import time


def python_list_addition(list1, list2):
    """Element-wise addition using Python lists."""
    return [a + b for a, b in zip(list1, list2)]


def python_list_multiplication(list1, list2):
    """Element-wise multiplication using Python lists."""
    return [a * b for a, b in zip(list1, list2)]


def python_list_sum(arr):
    """Sum of elements using Python lists."""
    return sum(arr)


def numpy_add(a, b):
    return a + b


def numpy_multiply(a, b):
    return a * b


def numpy_sum(a):
    return np.sum(a)



def benchmark_operation(name, python_func, numpy_func, python_args, numpy_args):
    """Benchmark a Python function against its NumPy equivalent."""
    # Warm-up runs
    _ = python_func(*python_args)
    _ = numpy_func(*numpy_args)
    
    # Benchmark Python version
    start_time = time.perf_counter()
    for _ in range(100):
        result_python = python_func(*python_args)
    python_time = time.perf_counter() - start_time
    
    # Benchmark NumPy version
    start_time = time.perf_counter()
    for _ in range(100):
        result_numpy = numpy_func(*numpy_args)
    numpy_time = time.perf_counter() - start_time
    
    speedup = python_time / numpy_time if numpy_time > 0 else float('inf')
    
    print(f"\n{name}:")
    print(f"  Python list time: {python_time:.6f} seconds (100 iterations)")
    print(f"  NumPy array time: {numpy_time:.6f} seconds (100 iterations)")
    print(f"  Speedup: {speedup:.2f}x faster with NumPy")
    
    return python_time, numpy_time, speedup


def run_benchmarks(size: int) -> None:
    """Create test data and run all benchmark comparisons."""
    print(f"\n{'=' * 70}")
    print(f"Array Size: {size:,} elements")
    print(f"{'=' * 70}")

    python_list1 = [float(i) for i in range(size)]
    python_list2 = [float(i * 2) for i in range(size)]
    numpy_array1 = np.array(python_list1)
    numpy_array2 = np.array(python_list2)

    benchmark_operation(
        "Element-wise Addition",
        python_list_addition,
        numpy_add,
        (python_list1, python_list2),
        (numpy_array1, numpy_array2),
    )

    benchmark_operation(
        "Element-wise Multiplication",
        python_list_multiplication,
        numpy_multiply,
        (python_list1, python_list2),
        (numpy_array1, numpy_array2),
    )

    benchmark_operation(
        "Sum of Elements",
        python_list_sum,
        numpy_sum,
        (python_list1,),
        (numpy_array1,),
    )



def main():
    """Main function to run all benchmarks."""
    print("=" * 70)
    print("NumPy vs Python Lists Performance Comparison")
    print("=" * 70)

    run_benchmarks(100000)

    print(f"\n{'=' * 70}")
    print("Summary:")
    print("=" * 70)
    print("NumPy is significantly faster than Python lists because:")
    print("  1. NumPy arrays are stored in contiguous memory blocks")
    print("  2. NumPy operations are implemented in C/C++/Fortran")
    print("  3. NumPy uses vectorized operations (SIMD instructions)")
    print("  4. NumPy avoids Python's overhead (type checking, etc.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
