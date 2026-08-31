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
// chunk size
static double dot_omp_tasks(const std::vector<double>& a,
                            const std::vector<double>& b,
                            std::size_t chunk_size) {
    const std::size_t n = a.size();
    double sum = 0.0;

    // taskgroup > groups the tasks below and waits for all of them
    // to finish before exiting the block
    
    // task_reduction(+:sum) > declares sum as a thread-safe reduction
    // target for the tasks created inside this region
    
    #pragma omp taskgroup task_reduction(+:sum)
    {
        for (std::size_t start = 0; start < n; start += chunk_size) { // from start by chunk_size
            const std::size_t end = std::min(start + chunk_size, n); // prevents last chunk extending 
            // one task per chunk:
            // firstprivate(start, end) > each task gets its own copy of the range
            // shared(a, b) > tasks read from the same original vectors
            // in_reduction(+:sum) > this task contributes to the sum declared above
            #pragma omp task firstprivate(start, end) shared(a, b) \
                              in_reduction(+:sum) default(none)
            {
                double partial = 0.0;
                // sum element-wise products for this chunk
                for (std::size_t i = start; i < end; ++i) {
                    partial += a[i] * b[i];
                }
                sum += partial;
            }
        }
    } 
    return sum;
}

// l2 chunk
static double l2_norm_omp_tasks(const std::vector<double>& x,
                                std::size_t chunk_size) {
    return std::sqrt(dot_omp_tasks(x, x, chunk_size)); // l2 norm = sqrt(x · x), reusing dot_omp_tasks
}

// normalize vec chunk
// vec x
// chunk size
static void normalize_omp_tasks(std::vector<double>& x,
                                std::size_t chunk_size) {

    const double nrm = l2_norm_omp_tasks(x, chunk_size); // l2 norm of the vec
    const double inv = 1.0 / nrm; // inverse of the norm
    const std::size_t n = x.size(); // vec dim

    // taskgroup > groups the tasks below and waits for all of them
    // to finish before exiting the block
    #pragma omp taskgroup
    {
        for (std::size_t start = 0; start < n; start += chunk_size) { // from start by chunk_size
            const std::size_t end = std::min(start + chunk_size, n); // prevents last chunk extending
            // one task per chunk:
            // firstprivate(start, end, inv) > each task gets its own copy
            // shared(x) > tasks write to the same vector but on disjoint ranges
            #pragma omp task firstprivate(start, end, inv) shared(x) \
                              default(none)
            {
                for (std::size_t i = start; i < end; ++i) {
                    x[i] *= inv; // scale each element in this chunk
                }
            }
        }
    } 
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
static void spmv_csr_shifted_rows_omp(const CSRMatrix& A,
                           std::size_t row_shift,
                           const std::vector<double>& x,
                           std::vector<double>& y,
                           std::size_t chunk_size) {
    const std::size_t n = A.n;

    // no pragma 
    for (std::size_t start = 0; start < n; start += chunk_size) { // from start by chunk_size
        const std::size_t end = std::min(start + chunk_size, n); // prevents last chunk extending

        // one task per chunk:
        // firstprivate(start, end, row_shift, n) > each task gets its own copy
        // shared(A, x, y) > tasks read A/x and write y on disjoint ranges 
        #pragma omp task firstprivate(start, end, row_shift, n) \
                          shared(A, x, y) default(none)
        {
            for (std::size_t i = start; i < end; ++i) {
                // physical row of the original matrix given the current shift
                const std::size_t src_row = (i + n - row_shift) % n; 

                double sum = 0.0;
                for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) { // CSR iter only the nonzeros values
                    sum += A.values[p] * x[A.col_idx[p]]; // multiply the nonzeros of the matrix row by the corresponding elements of the vector
                }
                y[i] = sum; // save the total in the correct spot
            }
        }
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
    
    // init the total displacement offset
    std::size_t row_shift = 0;

    // parallel > creates a team of threads that will execute the block below
    // num_threads > sets the team size to num_threads
    // shared > all listed variables are shared across the team
    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(A, x, y, n, shift_rows, chunk_size, norm_chunk_size, \
               row_shift, result, final_vector, timers, seed)
    {
        // only one thread of the team executes this block
        // the others stay idle ready to run the tasks it creates
        #pragma omp single
        {
            // init vec x with pseudo casual values
            auto t_init0 = get_time_now();
            SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
            for (double& v : x) {
                v = rng.next_unit();
            }

            // normalize x
            normalize_omp_tasks(x, norm_chunk_size);

            // stop the init timer
            timers.init_sec = get_elapsed_time(t_init0);

            // start the main loop 
            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                // check if matrix need to evolve
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    auto t_ep0 = get_time_now();
                    row_shift = (row_shift + shift_rows) % n;
                    timers.epoch_transition_sec += get_elapsed_time(t_ep0);
                }

                auto t_spmv0 = get_time_now();
                spmv_csr_shifted_rows_omp(A, row_shift, x, y, chunk_size); // call the SPMV creates tasks, does not wait
                // taskwait > block here until all spmv tasks above have finished
                #pragma omp taskwait
                timers.spmv_sec += get_elapsed_time(t_spmv0);

                auto t_vec0 = get_time_now();
                normalize_omp_tasks(y, norm_chunk_size); // call norm
                timers.vector_ops_sec += get_elapsed_time(t_vec0);

                x.swap(y); // swap the x vector to y
            }

            // Rayleigh requires multiplying the final vector by the matrix in its current state
            auto t_spmv_final = get_time_now();
            spmv_csr_shifted_rows_omp(A, row_shift, x, y, chunk_size);
            #pragma omp taskwait // taskwait > block here until all spmv tasks above have finished
            timers.spmv_sec += get_elapsed_time(t_spmv_final);

            // computes Raylight
            auto t_vec_final = get_time_now();
            result.rayleigh = dot_omp_tasks(x, y, norm_chunk_size);
            timers.vector_ops_sec += get_elapsed_time(t_vec_final);

            // computes checksum
            auto t_chk0 = get_time_now();
            result.checksum = checksum_vector(x);
            timers.vector_ops_sec += get_elapsed_time(t_chk0);
            result.final_row_shift = row_shift;

            // save vec
            if (final_vector != nullptr) {
                *final_vector = std::move(x);
            }
        } 
    } 

    // stop timers
    timers.total_sec = get_elapsed_time(t_start_total);
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

    std::cout << "SPARSE_ITERATION_OMP_TASKS\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";

    try {
        // matrix generation
        auto tg0 = get_time_now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const double generation_sec = get_elapsed_time(tg0);

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        // prepare final_vector_out pointer
        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // start the func
        OmpIterativeResult out = iterative_spmv_evolving(
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