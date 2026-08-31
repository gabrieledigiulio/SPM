#include "threadpool.hpp"
#include "matrix_generation.hpp"
#include "utils.hpp"

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

struct ThreadPoolIterativeResult {
    IterativeResult result;
    ExecutionTimers timers;
};

// dot chunk product
// vec a and b 
// reference to the thread pool
// chunk size
static double dot_parallel(const std::vector<double>& a, 
                           const std::vector<double>& b, 
                           ThreadPool& pool, 
                           std::size_t chunk_size) {

    const std::size_t n = a.size(); // vec dim
    std::vector<std::future<double>> futures; // vec future for each chunk
    
    futures.reserve((n + chunk_size - 1) / chunk_size); // pre alloc

    for (std::size_t start = 0; start < n; start += chunk_size) { // from start by chunk_size
        std::size_t end = std::min(start + chunk_size, n); // prevents last chunk extending 

        // future returned by enqueue is saved
        // lambda func
        futures.push_back(pool.enqueue([&a, &b, start, end]() { // for each chunk submit a new task to the pool
            double partial_sum = 0.0;
            // it sums the element-wise product only within the range [start, end) 
            // returns the partial sum which will end up in the corresponding future
            for (std::size_t i = start; i < end; ++i) {
                partial_sum += a[i] * b[i];
            }
            return partial_sum; 
        }));
    }

    double total_sum = 0.0;
    for (auto& fut : futures) { // for each future in the vector
        // it blocks until the corresponding task 
        // has finished calculating its partial sum then returns it
        total_sum += fut.get();
    }
    return total_sum;
}

// l2 chunk
static double l2_norm_parallel(const std::vector<double>& x, 
                               ThreadPool& pool, 
                               std::size_t chunk_size) {
    return std::sqrt(dot_parallel(x, x, pool, chunk_size)); // l2 norm = sqrt(x · x), reusing dot_parallel
}

// normalize vec chunk
// vec x
// reference to the thread pool
// chunk size
static void normalize_parallel(std::vector<double>& x, 
                               ThreadPool& pool, 
                               std::size_t chunk_size) {

    const double nrm = l2_norm_parallel(x, pool, chunk_size); // l2 norm of the vec
    const double inv = 1.0 / nrm; // inverse of the norm
    const std::size_t n = x.size(); // vec dim

    // tasks do not return anything but future is still needed for synch
    std::vector<std::future<void>> futures;
    
    futures.reserve((n + chunk_size - 1) / chunk_size); // pre alloc

    for (std::size_t start = 0; start < n; start += chunk_size) { // from start by chunk_size
        std::size_t end = std::min(start + chunk_size, n); // prevents last chunk extending
        // future returned by enqueue is saved
        // lambda func
        futures.push_back(pool.enqueue([&x, inv, start, end]() {  // for each chunk submit a new task to the pool
            // inside lambda iter only over the i assigned to this specific chunk
            for (std::size_t i = start; i < end; ++i) {
                x[i] *= inv; // divides it by the norm
            }
        }));
    }
    // on each element future
    for (auto& fut : futures) {
        fut.get(); // call get on each future
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
// reference to the thread pool
// chunk size
static void spmv_csr_shifted_rows_parallel(const CSRMatrix& A,
                                           std::size_t row_shift,
                                           const std::vector<double>& x,
                                           std::vector<double>& y,
                                           ThreadPool& pool,
                                           std::size_t chunk_size) {

    const std::size_t n = A.n; // vec dim
    std::vector<std::future<void>> futures; // vec future for each chunk
    
    futures.reserve((n + chunk_size - 1) / chunk_size); // pre alloc

    for (std::size_t start = 0; start < n; start += chunk_size) { // from start by chunk_size
        std::size_t end = std::min(start + chunk_size, n); // prevents last chunk extending
        // future returned by enqueue is saved 
        // lambda func
        futures.push_back(pool.enqueue([&A, row_shift, &x, &y, start, end, n]() { // for each chunk submit a new task to the pool
            // inside lambda iter only over the i assigned to this specific chunk
            for (std::size_t i = start; i < end; ++i) {
                // which physical row of the original matrix given the current shift
                const std::size_t src_row = (i + n - row_shift) % n;

                double sum = 0.0;
                for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) { // CSR iter only the nonzeros values
                    sum += A.values[p] * x[A.col_idx[p]];  // multiply the nonzeros of the matrix row by the corresponding elements of the vector
                }
                y[i] = sum; // save the total in the correct spot
            }
        }));
    }

    // on each element future
    for (auto& fut : futures) {
        fut.get();  // call get on each future
    }
}

// coordinator
// matrix A
// seed 
// reference to the thread pool
// chunk size
// norm chunk size
// optional pointer where the result is to be stored
static ThreadPoolIterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                               std::uint64_t seed,
                                               ThreadPool& pool,
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

    // init vec x with pseudo casual values
    auto t0 = get_time_now();
    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }

    // normalize x
    normalize_parallel(x, pool, norm_chunk_size); 

    // stop the init timer
    timers.init_sec = get_elapsed_time(t0);

    // init the total displacement offset
    std::size_t row_shift = 0;

    // start the main loop 
    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        // check if matrix need to evolve
        if (iter > 0 && (iter % EPOCH_LEN) == 0) {
            t0 = get_time_now();
            row_shift = (row_shift + shift_rows) % n;
            timers.epoch_transition_sec += get_elapsed_time(t0);
        }

        t0 = get_time_now();
        spmv_csr_shifted_rows_parallel(A, row_shift, x, y, pool, chunk_size); // call the SPMV 
        timers.spmv_sec += get_elapsed_time(t0);

        t0 = get_time_now();
        normalize_parallel(y, pool, norm_chunk_size); // call norm
        timers.vector_ops_sec += get_elapsed_time(t0);

        x.swap(y); // swap the x vector to y
    }

    // Rayleigh requires multiplying the final vector by the matrix in its current state
    t0 = get_time_now();
    spmv_csr_shifted_rows_parallel(A, row_shift, x, y, pool, chunk_size);
    timers.spmv_sec += get_elapsed_time(t0);

    // computes Raylight
    t0 = get_time_now();
    result.rayleigh = dot_parallel(x, y, pool, norm_chunk_size); 
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
    
    return ThreadPoolIterativeResult{result, timers};
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
        ? chunk_size 
        : static_cast<std::size_t>(norm_chunk_arg);

    std::cout << "SPARSE_ITERATION_CPP_THREADS\n";
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
        
        // create pool
        ThreadPool pool(num_threads);

        // start the func
        ThreadPoolIterativeResult out = iterative_spmv_evolving(
            G.A, seed, pool, chunk_size, norm_chunk_size, final_vector_out);
        
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