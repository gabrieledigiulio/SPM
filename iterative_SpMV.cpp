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


// collect output data

struct IterativeResult {
  double rayleigh = 0.0;
  std::uint64_t checksum = 0;
  std::size_t final_row_shift = 0;
};

struct SeqIterativeResult {
    IterativeResult result;
    ExecutionTimers timers;
};


// dot product

static double dot(const std::vector<double> &a, const std::vector<double> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// l2 norm

static double l2_norm(const std::vector<double> &x) {
  return std::sqrt(dot(x, x));
}

// vectorn normalization

static void normalize(std::vector<double> &x) {
  const double nrm = l2_norm(x);
  const double inv = 1.0 / nrm;

  for (double &v : x) {
    v *= inv;
  }
}

// compute shift 

static std::size_t compute_shift_rows(std::size_t n) {
  std::size_t s = n / 16 + 17;
  if ((s % 2) == 0)
    ++s;
  s %= n;
  if (s == 0)
    s = 1;
  return s;
}


// multiplication of the sparse matrix A and the vector x
// matrix A
// current shift value row_shift
// input vector x
// target vector y

static void spmv_csr_shifted_rows(const CSRMatrix &A, std::size_t row_shift,
                                  const std::vector<double> &x,
                                  std::vector<double> &y) {
  const std::size_t n = A.n; // takes dim
  y.assign(n, 0.0); // init full of zeros

  for (std::size_t i = 0; i < n; ++i) { // for every rows of y
    const std::size_t src_row = (i + n - row_shift) % n; // physical row for logical row i after the shift

    double sum = 0.0; // init accumulator

    for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; // CSR iter only the nonzeros values
         ++p) {
      sum += A.values[p] * x[A.col_idx[p]]; // multiply the nonzeros of the matrix row by the corresponding elements of the vector
    }

    y[i] = sum; // save the total in the correct spot
  }
}

// coordinator
// the matrix A
// the seed 
// optional pointer where the result is to be stored

static SeqIterativeResult iterative_spmv_evolving(const CSRMatrix &A, std::uint64_t seed,
                        std::vector<double> *final_vector = nullptr) {
  
  // creates obj for stats
  ExecutionTimers timers;
  IterativeResult result;
  
  const std::size_t n = A.n;  // init dim
  const std::size_t shift_rows = compute_shift_rows(n); // how many rows to shift per epoch
  
  // alloc vec
  std::vector<double> x(n);
  std::vector<double> y(n);
  
  // start global timer
  const auto t_start_total = get_time_now();
  
  // init vec x with pseudo casual values
  auto t0 = get_time_now();
  SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
  for (double &v : x) {
    v = rng.next_unit();
  }

  // normalize x
  normalize(x);

  // stop the init timer
  timers.init_sec = get_elapsed_time(t0);

  // init the total displacement offset
  std::size_t row_shift = 0;

  // start the main loop 
  for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
    // check if matrix need to evolve
    if (iter > 0 && (iter % EPOCH_LEN) == 0) {
      t0 = get_time_now();
      row_shift = (row_shift + shift_rows) % n; // row shift
      timers.epoch_transition_sec += get_elapsed_time(t0); 
    }

    t0 = get_time_now();
    spmv_csr_shifted_rows(A, row_shift, x, y); // call the SPMV 
    timers.spmv_sec += get_elapsed_time(t0);

    t0 = get_time_now();
    normalize(y); // call norm
    timers.vector_ops_sec += get_elapsed_time(t0);

    x.swap(y); // swap the x vector to y
  }

  // Rayleigh requires multiplying the final vector by the matrix in its current state
  t0 = get_time_now();
  spmv_csr_shifted_rows(A, row_shift, x, y);
  timers.spmv_sec += get_elapsed_time(t0);

  // computes Raylight
  t0 = get_time_now();
  result.rayleigh = dot(x, y);
  timers.vector_ops_sec += get_elapsed_time(t0);
  
  // computes checksum
  t0 = get_time_now();
  result.checksum = checksum_vector(x);
  timers.vector_ops_sec += get_elapsed_time(t0);
  result.final_row_shift = row_shift;

  // save vec
  if (final_vector != nullptr) {
    *final_vector = std::move(x);
  }
    // stop timers
    timers.total_sec = get_elapsed_time(t_start_total);
    timers.computation_sec = timers.spmv_sec + timers.vector_ops_sec + timers.epoch_transition_sec;
    return SeqIterativeResult{result, timers};
}

int main(int argc, char **argv) {

  // inti var
  std::uint64_t n64 = 0;
  std::uint64_t nz = 0;
  std::uint64_t seed = 111;
  std::string mode;
  std::string dump_vector_path;

  // check args
  if (!read_arg_u64(argc, argv, "-n", n64) ||
      !read_arg_u64(argc, argv, "-nz", nz) ||
      !read_arg_str(argc, argv, "-m", mode)) {
    usage(argv[0]);
    return 1;
  }

  // optional args
  (void)read_arg_u64(argc, argv, "-s", seed);
  (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

  // read input as a wide integer then cast to the native type used for the matrix
  const std::size_t n = static_cast<std::size_t>(n64);
  std::cout << "SPARSE_ITERATION_SEQ\n";

  try {
    // matrix generation
    const auto tg0 = get_time_now();
    const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
    const auto tg1 = get_time_now();

    const double generation_sec =
        std::chrono::duration<double>(tg1 - tg0).count();

    print_matrix_stats(G);
    std::cout << "generation_time_sec=" << generation_sec << "\n\n";

    // prepare final_vector_out pointer
    std::vector<double> final_vector;
    std::vector<double> *final_vector_out =
        dump_vector_path.empty() ? nullptr : &final_vector; //check dump

    // start the func
    SeqIterativeResult out =
        iterative_spmv_evolving(G.A, seed, final_vector_out);

    // print results
    print_all_timers(out.timers);

    std::cout << std::setprecision(15);
    std::cout << "rayleigh=" << out.result.rayleigh << "\n";
    std::cout << "checksum=0x" << std::hex << out.result.checksum << std::dec << "\n";

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