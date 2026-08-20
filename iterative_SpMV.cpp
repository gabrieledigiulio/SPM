//
// Sequential reference implementation for the One-Shot project:
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// The code provides:
//   - generation of a sparse matrix in CSR format
//   - two sparsity modes:
//       regular   : nonzeros almost uniformly distributed across rows
//       irregular : nonzeros concentrated in dense row regions
//   - iterative sparse matrix-vector computation
//   - row evolution by circular row shifts at epoch boundaries
//   - correctness output through a checksum and optional final-vector dump
//
// Matrix generation is implemented in matrix_generation.hpp.
// Generic helper functions are implemented in utils.hpp.
//
// Evolution model:
//   The matrix is generated once. Every EPOCH_LEN iterations, its rows are
//   logically shifted by shift_rows positions. In the sequential code this is
//   handled without physically moving matrix rows: the SpMV kernel maps each
//   logical row to the corresponding source row in the original CSR matrix.
//
//   In a distributed implementation, however, the same evolution must be
//   reflected consistently in the distributed data layout.
//
// Command line:
//   -n  N        matrix size, NxN
//   -nz K        total number of nonzeros
//   -m  mode     regular | irregular
//   -s  seed     optional seed, default 111
//   --dump-vector FILE
//                 optional dump of the final normalized vector
//
// Minimal build:
//   g++ -O3 -std=c++20 -I . -Wall iterative_SpMV.cpp -o seq
//
// Examples:
//   ./seq -n 500000 -nz 20000000 -m regular
//   ./seq -n 500000 -nz 20000000 -m irregular
//   ./seq -n 5000 -nz 20000 -m irregular --dump-vector seq_vec.dump
//
// Notes:
//   - Matrix generation is not included in computation time.
//   - The computation uses a fixed number of iterations.
//   - The main workload is the irregular case.
//

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"

// number of iterations
static constexpr std::uint32_t NUM_ITERS = 500;
// number of iterations between two matrix-evolution steps
static constexpr std::uint32_t EPOCH_LEN = 25;

// ==========================================
// 1. STRUTTURE PER TELEMETRIA
// ==========================================
struct ExecutionTimers {
    double init_sec              = 0.0;
    double spmv_sec              = 0.0;
    double vector_ops_sec        = 0.0;
    double epoch_transition_sec  = 0.0;
    double total_sec             = 0.0;
};

struct IterativeResult {
  double rayleigh = 0.0;
  std::uint64_t checksum = 0;
  std::size_t final_row_shift = 0;
};

struct SeqIterativeResult {
    IterativeResult result;
    ExecutionTimers timers;
};

static double get_elapsed(const std::chrono::steady_clock::time_point& start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ==========================================
// 2. OPERAZIONI VETTORIALI
// ==========================================

static double dot(const std::vector<double> &a, const std::vector<double> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static double l2_norm(const std::vector<double> &x) {
  return std::sqrt(dot(x, x));
}

static void normalize(std::vector<double> &x) {
  const double nrm = l2_norm(x);
  const double inv = 1.0 / nrm;

  for (double &v : x) {
    v *= inv;
  }
}

// Computes the epoch parameter
static std::size_t compute_shift_rows(std::size_t n) {
  std::size_t s = n / 16 + 17;
  if ((s % 2) == 0)
    ++s;
  s %= n;
  if (s == 0)
    s = 1;
  return s;
}

// Per-row SpMV kernel.
static void spmv_csr_shifted_rows(const CSRMatrix &A, std::size_t row_shift,
                                  const std::vector<double> &x,
                                  std::vector<double> &y) {
  const std::size_t n = A.n;
  y.assign(n, 0.0);

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t src_row = (i + n - row_shift) % n;

    double sum = 0.0;
    for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1];
         ++p) {
      sum += A.values[p] * x[A.col_idx[p]];
    }

    y[i] = sum;
  }
}

// ==========================================
// 3. FUNZIONE ITERATIVA
// ==========================================

static SeqIterativeResult
iterative_spmv_evolving(const CSRMatrix &A, std::uint64_t seed,
                        std::vector<double> *final_vector = nullptr) {
  ExecutionTimers timers;
  IterativeResult result;

  const std::size_t n = A.n;
  const std::size_t shift_rows = compute_shift_rows(n);

  std::vector<double> x(n);
  std::vector<double> y(n);

  const auto t_start_total = std::chrono::steady_clock::now();

  // Fase 1: Inizializzazione (RNG + normalizzazione)
  auto t0 = std::chrono::steady_clock::now();
  SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
  for (double &v : x) {
    v = rng.next_unit();
  }
  normalize(x);
  timers.init_sec = get_elapsed(t0);

  std::size_t row_shift = 0;

  for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
    if (iter > 0 && (iter % EPOCH_LEN) == 0) {
      t0 = std::chrono::steady_clock::now();
      row_shift = (row_shift + shift_rows) % n;
      timers.epoch_transition_sec += get_elapsed(t0);
    }

    t0 = std::chrono::steady_clock::now();
    spmv_csr_shifted_rows(A, row_shift, x, y);
    timers.spmv_sec += get_elapsed(t0);

    t0 = std::chrono::steady_clock::now();
    normalize(y);
    timers.vector_ops_sec += get_elapsed(t0);

    x.swap(y);
  }

  t0 = std::chrono::steady_clock::now();
  spmv_csr_shifted_rows(A, row_shift, x, y);
  timers.spmv_sec += get_elapsed(t0);
  
  t0 = std::chrono::steady_clock::now();
  result.rayleigh = dot(x, y);
  timers.vector_ops_sec += get_elapsed(t0);
  
  t0 = std::chrono::steady_clock::now();
  result.checksum = checksum_vector(x);
  timers.vector_ops_sec += get_elapsed(t0);
  result.final_row_shift = row_shift;

  if (final_vector != nullptr) {
    *final_vector = std::move(x);
  }

  timers.total_sec = get_elapsed(t_start_total);

  return SeqIterativeResult{result, timers};
}

int main(int argc, char **argv) {
  std::uint64_t n64 = 0;
  std::uint64_t nz = 0;
  std::uint64_t seed = 111;
  std::string mode;
  std::string dump_vector_path;

  if (!read_arg_u64(argc, argv, "-n", n64) ||
      !read_arg_u64(argc, argv, "-nz", nz) ||
      !read_arg_str(argc, argv, "-m", mode)) {
    usage(argv[0]);
    return 1;
  }

  (void)read_arg_u64(argc, argv, "-s", seed);
  (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

  const std::size_t n = static_cast<std::size_t>(n64);
  std::cout << "SPARSE_ITERATION_SEQ\n";

  try {
    const auto tg0 = std::chrono::steady_clock::now();
    const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
    const auto tg1 = std::chrono::steady_clock::now();

    const double generation_sec =
        std::chrono::duration<double>(tg1 - tg0).count();

    print_matrix_stats(G);
    std::cout << "generation_time_sec=" << generation_sec << "\n\n";

    std::vector<double> final_vector;
    std::vector<double> *final_vector_out =
        dump_vector_path.empty() ? nullptr : &final_vector;

    const SeqIterativeResult out =
        iterative_spmv_evolving(G.A, seed, final_vector_out);

    std::cout << "Time breakdown (seconds):\n";
    std::cout << "  SpMV time (sec) = " << out.timers.spmv_sec << "\n";
    std::cout << "  Vector ops time (sec) = " << out.timers.vector_ops_sec << "\n";
    std::cout << "  Epoch transition (sec) = " << out.timers.epoch_transition_sec << "\n";
    std::cout << "  Init time (sec) = " << out.timers.init_sec << "\n";

    std::cout << std::setprecision(15);
    std::cout << "Computation time (sec) = " << out.timers.total_sec << "\n";
    std::cout << "rayleigh=" << out.result.rayleigh << "\n";
    std::cout << "checksum=0x" << std::hex << out.result.checksum << std::dec
              << "\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Time (sec) = " << out.timers.total_sec << "\n";

    if (!dump_vector_path.empty()) {
      dump_vector(dump_vector_path, final_vector);
      std::cout << "vector_dump=" << dump_vector_path << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}