# Evolving SpMV: Parallel Iterative Sparse Matrix-Vector Multiplication

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![OpenMP](https://img.shields.io/badge/OpenMP-5.0+-green.svg)](https://www.openmp.org/)
[![MPI](https://img.shields.io/badge/MPI-PMIx%20%2F%20Slurm-orange.svg)](https://www.open-mpi.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)

This project explores and compares different parallelization strategies for iterative **Sparse Matrix-Vector Multiplication (SpMV)** on a sparse matrix that evolves over time via circular row shifts. It benchmarks and analyzes performance, load balancing, granularity, and communication overhead across shared-memory (C++ Threads, OpenMP) and distributed-memory (MPI + OpenMP) architectures.


## Parallel Implementations

| Implementation | Source File | Paradigm & Technology | Description |
| :--- | :--- | :--- | :--- |
| **Sequential** | [`iterative_SpMV.cpp`](iterative_SpMV.cpp) | Single-Threaded C++20 | Baseline reference implementation with zero synchronization overhead. |
| **C++ Thread Pool** | [`threadpool_SpMV.cpp`](threadpool_SpMV.cpp) | C++20 Thread Pool & Tasks | Modern task pool leveraging `std::jthread`, `std::future`, and concurrent worker queues. |
| **OpenMP Tasks** | [`omp_tasks_SpMV.cpp`](omp_tasks_SpMV.cpp) | OpenMP Tasking | Dynamic task graph scheduling using `#pragma omp task` and `#pragma omp taskgroup`. |
| **OpenMP Work-Sharing** | [`omp_worksharing_SpMV.cpp`](omp_worksharing_SpMV.cpp) | OpenMP Work-Sharing | Thread-team parallelization using static `#pragma omp for schedule(static, chunk_size)`. |
| **MPI + OpenMP Tasks** | [`mpi_omp_tasks_SpMV.cpp`](mpi_omp_tasks_SpMV.cpp) | Distributed Hybrid Tasks | Non-zero distributed matrix decomposition + multi-threaded local OpenMP task execution. |
| **MPI + OpenMP Work-Sharing** | [`mpi_omp_worksharing_SpMV.cpp`](mpi_omp_worksharing_SpMV.cpp) | Distributed Hybrid Loops | Distributed memory decomposition + OpenMP static work-sharing per rank. |

---

## Problem & Computational Workflow

### The Computational Loop
At each iteration $k \in [0, 500)$, the program repeatedly performs a Sparse Matrix-Vector multiplication followed by an $L_2$ vector normalization:

1. **Matrix Evolution (every 25 iterations)**: The matrix undergoes a circular row shift:
   $$\mathrm{src\_row} = (i + N - \mathrm{row\_shift}) \pmod N$$
   This alters the active non-zero distribution across threads and nodes dynamically, without requiring costly matrix reallocations.
2. **SpMV Multiplication**: Computes the intermediate dense vector $y = A_{\text{shifted}} \cdot x$.
3. **Normalization**: Calculates the Euclidean norm $\|y\|_2 = \sqrt{y \cdot y}$ and normalizes $y = y / \|y\|_2$.
4. **Pointer Swap**: Swaps vectors ($x \leftrightarrow y$) to prepare for the next step.

### Validation & Diagnostics
After completing 500 iterations, the program computes two invariant metrics to verify numerical correctness across all parallel variants:
* **Rayleigh Quotient**: $\lambda = \frac{x^T A_{\text{shifted}} x}{x^T x} = x^T y$, representing the dominant eigenvalue estimate.
* **64-bit Bitwise Checksum**: A deterministic hash of the final IEEE-754 vector elements to ensure bit-level parity across parallel runs.

### Sparsity Patterns
To stress-test different parallelization strategies, matrices can be generated in two modes:
* **`regular`**: Non-zero elements are uniformly distributed across all rows, providing an evenly balanced workload for static chunking.
* **`irregular`**: Non-zeros are concentrated in dense bands, creating severe computational imbalance that stresses dynamic task scheduling and MPI non-zero domain decomposition.

---

## Project Structure

```
SPM/
├── iterative_SpMV.cpp             # Sequential baseline
├── threadpool_SpMV.cpp            # Custom C++20 thread pool implementation
├── omp_tasks_SpMV.cpp             # OpenMP task-based implementation
├── omp_worksharing_SpMV.cpp       # OpenMP work-sharing implementation
├── mpi_omp_tasks_SpMV.cpp         # Hybrid MPI + OpenMP tasks implementation
├── mpi_omp_worksharing_SpMV.cpp   # Hybrid MPI + OpenMP work-sharing implementation
├── matrix_generation.hpp          # CSR matrix generator (regular & irregular sparsity)
├── utils.hpp                      # Core utilities: CLI parsing, timers, checksum, diagnostics
├── threadpool.hpp                 # Lock-based concurrent worker thread pool
├── plot.py                        # Centralized visualization and figure generation script
├── README.md                      # Project documentation
│
├── exp/                           # Benchmarking and experimentation suite
│   ├── run_seq.sh                 # Sequential baseline evaluation
│   ├── run_correctness_test.sh    # Numerical cross-validation across all paradigms
│   ├── run_omp_task_vs_work.sh    # OpenMP Tasks vs Work-Sharing benchmark
│   ├── run_mpi_task_vs_work.sh    # MPI+OpenMP Tasks vs Work-Sharing benchmark
│   ├── run_mpi_sweep.sh           # MPI Ranks vs OpenMP Threads grid exploration
│   ├── run_granularity.sh         # SpMV task chunk size sweep
│   ├── run_granularity_norm.sh    # Vector normalization chunk size sweep
│   ├── run_regular_irregular.sh   # Memory access pattern comparison
│   ├── run_strong_scalability.sh  # Multi-node strong scaling evaluation
│   ├── run_weak_scalability.sh    # Proportional weak scaling evaluation
│   ├── run_weak_scalability_constant.sh # Weak scaling with constant matrix dimension N
│   └── results/                   # Experimental CSV data output directory
│
└── img/                           # Generated benchmark charts and visual figures
```

---

## Prerequisites & Dependencies

* **C++ Compiler**: GCC 10+, Clang 11+, or any toolchain fully supporting **C++20**.
* **OpenMP**: OpenMP 4.5+ or 5.0+ runtime support.
* **MPI Implementation**: OpenMPI 4.0+ or MPICH with `mpicxx` wrapper.
* **Workload Manager**: Slurm environment (`srun`) for cluster-scale execution.
* **Python 3** (for analysis & plotting):
  ```bash
  pip install pandas matplotlib numpy
  ```

---

## Build Instructions

Compile the binaries directly from the project root directory:

```bash
# 1. Sequential Reference
g++ -O3 -std=c++20 -I . -Wall -Wextra iterative_SpMV.cpp -o iterative_SpMV

# 2. C++ Thread Pool
g++ -O3 -std=c++20 -I . -Wall -Wextra -pthread threadpool_SpMV.cpp -o threadpool_SpMV

# 3. OpenMP Tasks
g++ -O3 -std=c++20 -I . -Wall -Wextra -fopenmp omp_tasks_SpMV.cpp -o omp_tasks_SpMV

# 4. OpenMP Work-Sharing
g++ -O3 -std=c++20 -I . -Wall -Wextra -fopenmp omp_worksharing_SpMV.cpp -o omp_worksharing_SpMV

# 5. Hybrid MPI + OpenMP Tasks
mpicxx -O3 -std=c++20 -I . -Wall -Wextra -fopenmp mpi_omp_tasks_SpMV.cpp -o mpi_omp_tasks_SpMV

# 6. Hybrid MPI + OpenMP Work-Sharing
mpicxx -O3 -std=c++20 -I . -Wall -Wextra -fopenmp mpi_omp_worksharing_SpMV.cpp -o mpi_omp_worksharing_SpMV
```

---

## CLI & Execution Reference

### Command-Line Arguments

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-n` | `uint64` | Matrix dimension ($N \times N$) | *Mandatory* |
| `-nz` | `uint64` | Total number of non-zero elements ($NZ$) | *Mandatory* |
| `-m` | `string` | Sparsity distribution mode: `regular` or `irregular` | *Mandatory* |
| `-t` | `uint64` | Number of worker threads (or threads per MPI rank) | *Mandatory (parallel)* |
| `-c` | `uint64` | SpMV chunk size (number of rows per parallel task/chunk) | *Mandatory (parallel)* |
| `-nc` | `uint64` | Vector normalization chunk size | *Mandatory (parallel)* |
| `-s` | `uint64` | Deterministic random seed for matrix and PRNG | `111` |
| `--dump-vector` | `string` | Output filepath to dump final normalized vector | *None (disabled)* |

### Example Executions

#### Standalone Shared-Memory Run
```bash
# Run OpenMP Task-Based SpMV with 16 threads and chunk size 2048
./omp_tasks_SpMV -n 1000000 -nz 250000000 -m irregular -t 16 -c 2048 -nc 2048 -s 111
```

#### Distributed Cluster Run (Slurm + MPI)
```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=16

# Run 8 nodes, 1 rank/node, 16 OpenMP threads per rank
srun --time=00:15:00 --mpi=pmix -N 8 -n 8 --cpus-per-task=16 \
    ./mpi_omp_tasks_SpMV -n 1000000 -nz 250000000 -m irregular -t 16 -c 2048 -nc 2048 -s 111
```

---

## Benchmark Suite (`exp/`)

All benchmark scripts are fully automated, calculate median metrics across repeated runs, and export standardized CSV outputs directly into `exp/results/`:

```bash
cd exp/
chmod +x *.sh

# Execute any desired experiment
./run_strong_scalability.sh
```

### Experiment Inventory

| Benchmark Script | Purpose & Description | Output CSV |
| :--- | :--- | :--- |
| [`run_seq.sh`](exp/run_seq.sh) | Evaluates baseline single-threaded sequential performance. | `sequential_results.csv` |
| [`run_correctness_test.sh`](exp/run_correctness_test.sh) | Cross-validates Rayleigh quotient and checksum across all implementations. | `cross_validation_results.csv` |
| [`run_regular_irregular.sh`](exp/run_regular_irregular.sh) | Quantifies the performance impact of regular vs. irregular sparsity structures. | `regular_vs_irregular.csv` |
| [`run_omp_task_vs_work.sh`](exp/run_omp_task_vs_work.sh) | Compares OpenMP Tasks vs. Work-Sharing on single-node (4 to 32 threads). | `omp_task_vs_work.csv` |
| [`run_mpi_task_vs_work.sh`](exp/run_mpi_task_vs_work.sh) | Compares MPI+OMP Tasks vs. Work-Sharing on clusters (1 to 8 nodes). | `mpi_task_vs_work.csv` |
| [`run_strong_scalability.sh`](exp/run_strong_scalability.sh) | Evaluates strong scalability ($N=1\text{M}, NZ=250\text{M}$) across 1, 2, 4, 8 nodes. | `strong_scaling_results.csv` |
| [`run_weak_scalability.sh`](exp/run_weak_scalability.sh) | Evaluates weak scalability scaling both $N$ and $NZ$ proportionally with nodes. | `weak_scaling_results.csv` |
| [`run_weak_scalability_constant.sh`](exp/run_weak_scalability_constant.sh) | Evaluates weak scalability with fixed $N$ and proportional non-zero density. | `weak_scaling_constant_results.csv` |
| [`run_granularity.sh`](exp/run_granularity.sh) | Sweeps SpMV chunk sizes (256 to 16,384 rows/chunk) to find optimal granularity. | `granularity_results.csv` |
| [`run_granularity_norm.sh`](exp/run_granularity_norm.sh) | Sweeps normalization chunk sizes with fixed SpMV chunk size. | `norm_granularity_results.csv` |
| [`run_mpi_sweep.sh`](exp/run_mpi_sweep.sh) | Explores hybrid MPI rank vs. OpenMP thread balance for fixed core budgets. | `mpi_sweep_results.csv` |

---

## Plotting & Visual Analytics (`plot.py`)

The centralized plotting engine reads experimental results from `exp/results/` and outputs high-resolution figures to `img/`:

```bash
# Generate all plots
python3 plot.py --all

# Generate specific plots
python3 plot.py --strong-scaling       # Strong scaling execution time and speedup
python3 plot.py --weak-scaling         # Weak scaling execution time and efficiency
python3 plot.py --breakdown            # Stacked phase breakdown (Computation, Comm, Reduction)
python3 plot.py --regular-vs-irregular # Memory access pattern comparison
python3 plot.py --task-vs-work         # Tasking vs Work-Sharing comparative scaling
```

*Developed for the Parallel and Distributed Systems course*
