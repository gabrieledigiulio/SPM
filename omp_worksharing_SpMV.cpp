#include "matrix_generation.hpp"
#include "utils.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

struct OmpIterativeResult {
    IterativeResult result;
    ExecutionTimers timers;
};

// dot chunk product 
// vec a and b 
// dim n
// chunk size
static double dot_omp_for(const std::vector<double>& a,
                          const std::vector<double>& b,
                          std::size_t n,
                          std::size_t chunk_size) {

    double sum = 0.0; // fresh per call acc
    
    // for > tells the existing thread team to divide the iterations of this loop among yourselves
    // schedule(static, chunk_size) > the strategy for assigning iterations to threads
    // reduction(+:sum) > each thread gets a private copy of sum, initialized to 0
    #pragma omp for schedule(static, chunk_size) reduction(+:sum)
    // each thread performs this summation only on 
    // the iterations assigned to it by the schedule
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i]; // sum element-wise products
    }
    // implicit barrier
    return sum;
}

// scales v in place by factor over [0, n) 
// vec v
// dim n
// factor the scalar by which to multiply each component
// chunk size
static void scale_omp_for(std::vector<double>& v,
                          std::size_t n,
                          double factor,
                          std::size_t chunk_size) {
    // for > tells the existing thread team to divide the iterations of this loop among yourselves
    // schedule(static, chunk_size) > the strategy for assigning iterations to threads
    #pragma omp for schedule(static, chunk_size)
    for (std::size_t i = 0; i < n; ++i) {
        v[i] *= factor;
    }
}

// normalizes v in place (L2 norm), over [0, n)
// vec v
// dim n
// chunk size
static void normalize_omp_for(std::vector<double>& v,
                              std::size_t n,
                              std::size_t chunk_size) {
    
    const double sumsq = dot_omp_for(v, v, n, chunk_size); // call dot

    // sumsq is already guaranteed to be identical across all threads
    // each thread is allowed to recalculate inv independently
    // the benefit is avoiding a synchronization point
    const double inv = 1.0 / std::sqrt(sumsq);

    // the team redistributes the iterations to apply v[i] *= inv
    scale_omp_for(v, n, inv, chunk_size);
}

// compute shift 

static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// multiplication of the sparse matrix A and the vector x
// matrix A
// current shift value row_shift
// input vector x
// target vector y
// chunk size
static void spmv_csr_shifted_rows_omp_for(const CSRMatrix& A,
                          std::size_t row_shift,
                          const std::vector<double>& x,
                          std::vector<double>& y,
                          std::size_t chunk_size) {
    const std::size_t n = A.n;

    // the iterations are not pre-assigned: there is a central queue of blocks
    // each consisting of chunk_size iterations and each thread dynamically requests another block 
    // from the queue as soon as it finishes the current one
    #pragma omp for schedule(dynamic, chunk_size)
    for (std::size_t i = 0; i < n; ++i) { //each thread dynamically receives subsets of values ​​to process.
        const std::size_t src_row = (i + n - row_shift) % n; // physical row of the original matrix given the current shift

        double sum = 0.0;
        for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) { // CSR iter only the nonzeros values
            sum += A.values[p] * x[A.col_idx[p]]; // multiply the nonzeros of the matrix row by the corresponding elements of the vector
        }
        y[i] = sum; // save the total in the correct spot
    }
}

// coordinator
// matrix A
// seed 
// n threads
// chunk size
// norm chunk size
// optional pointer where the result is to be stored
static OmpIterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                                   std::uint64_t seed,
                                                   std::size_t num_threads,
                                                   std::size_t chunk_size,
                                                   std::size_t norm_chunk_size,
                                                   std::vector<double>* final_vector = nullptr) {
    // creates obj for stats
    ExecutionTimers timers;
    IterativeResult result;

    const std::size_t n = A.n; // init dim
    const std::size_t shift_rows = compute_shift_rows(n); // how many rows to shift per epoch

    // alloc vec
    std::vector<double> x(n);
    std::vector<double> y(n);

    // start global timer
    const auto t_start_total = get_time_now();
    std::size_t row_shift = 0;

    // t0 must be shared because consecutive single blocks might be executed by different threads:
    // this prevents reading an uninitialized thread-local copy
    std::chrono::steady_clock::time_point t0;

    // thread team is created once for the entire duration of the function
    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(A, x, y, n, shift_rows, chunk_size, norm_chunk_size, \
               row_shift, result, timers, seed, final_vector, t0)
    {
        // only one thread of the team executes this block; the RNG has
        // sequential internal state, so filling x cannot be split across threads
        #pragma omp single
        {
            // init vec x with pseudo casual values
            t0 = get_time_now();
            SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
            for (double& v : x) {
                v = rng.next_unit();
            }
        } // every thread waits here until x is fully initialized

        // normalize x called by the whole team
        normalize_omp_for(x, n, norm_chunk_size);

        // record how long init took; all threads wait here until it's registered
        #pragma omp single
        { timers.init_sec = get_elapsed_time(t0); }

        // main loop
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            // every thread evaluates this identically (iter/EPOCH_LEN are shared
            // and read-only here), so all threads agree on whether to enter
            if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                // the actual update of row_shift is isolated within a single,
                // since it's a write to a shared variable
                #pragma omp single
                {
                    const auto t_ep0 = get_time_now();
                    row_shift = (row_shift + shift_rows) % n;
                    timers.epoch_transition_sec += get_elapsed_time(t_ep0);
                } // implicit barrier: everyone waits until row_shift is updated
            }

            // take t0 after the barrier above, so every thread enters
            // spmv_csr_shifted_rows_omp_for at the same instant
            #pragma omp single
            { t0 = get_time_now(); }

            spmv_csr_shifted_rows_omp_for(A, row_shift, x, y, chunk_size); // whole-team call

            // accumulate elapsed spmv time; all threads wait until it's registered
            #pragma omp single
            { timers.spmv_sec += get_elapsed_time(t0); }

            #pragma omp single
            { t0 = get_time_now(); }

            // call norm whole-team call, same as above
            normalize_omp_for(y, n, norm_chunk_size);

            #pragma omp single
            {
                timers.vector_ops_sec += get_elapsed_time(t0);
                x.swap(y); // swap done safely, by a single thread
            } // implicit barrier: no one starts the new iteration until the swap is done
        }

        // Rayleigh requires multiplying the final vector by the matrix in its current state
        #pragma omp single
        { t0 = get_time_now(); }

        spmv_csr_shifted_rows_omp_for(A, row_shift, x, y, chunk_size);

        #pragma omp single
        {
            timers.spmv_sec += get_elapsed_time(t0);
            t0 = get_time_now();
        }

        // reduction inside dot_omp_for guarantees rayleigh_local is identical
        // on every thread, so no single/broadcast is needed to use it below
        const double rayleigh_local = dot_omp_for(x, y, n, norm_chunk_size); // x · (A_shifted * x)

        #pragma omp single
        {
            // computes Rayleigh
            result.rayleigh = rayleigh_local;
            timers.vector_ops_sec += get_elapsed_time(t0);

            // computes checksum
            t0 = get_time_now();
            result.checksum = checksum_vector(x);
            result.final_row_shift = row_shift;

            // save vec  
            if (final_vector != nullptr) {
                *final_vector = std::move(x);
            }
            timers.vector_ops_sec += get_elapsed_time(t0);
        }
    } // end of persistent parallel region

    timers.total_sec = get_elapsed_time(t_start_total);
    // computation time is the sum of the components
    timers.computation_sec = timers.spmv_sec + timers.vector_ops_sec + timers.epoch_transition_sec;

    return OmpIterativeResult{result, timers};
}


int main(int argc, char** argv) {
    //init var
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t num_threads, chunk_size, norm_chunk_arg;
    std::string mode;
    std::string dump_vector_path;

    // check args
    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "-t", num_threads) ||
        !read_arg_u64(argc, argv, "-c", chunk_size) ||
        !read_arg_u64(argc, argv, "-nc", norm_chunk_arg)) {

        usage(argv[0]);
        return 1;
    }

    // optional args
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    // read input as a wide integer then cast to the native type used for the matrix
    const std::size_t n = static_cast<std::size_t>(n64);

    // reuse chunk_size otherwise use the user's explicit value
    const std::size_t norm_chunk_size = (norm_chunk_arg == 0)
        ? static_cast<std::size_t>(chunk_size)
        : static_cast<std::size_t>(norm_chunk_arg);

    std::cout << "SPARSE_ITERATION_OMP_WORKSHARING\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";

    try {
        // matrix generation
        const auto tg0 = get_time_now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const double generation_sec = get_elapsed_time(tg0);

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        // prepare final_vector_out pointer
        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // start the func
        const OmpIterativeResult out = iterative_spmv_evolving(
            G.A, seed, num_threads, chunk_size, norm_chunk_size, final_vector_out);

        // print results
        print_all_timers(out.timers);

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << out.result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << out.result.checksum << std::dec << "\n";

        if (!dump_vector_path.empty()) {
            dump_vector(dump_vector_path, final_vector);
            std::cout << "vector_dump=" << dump_vector_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}