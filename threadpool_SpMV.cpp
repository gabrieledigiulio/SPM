// ==============================================================================
// SPM "One-Shot" Project - C++ Threads Implementation
// ==============================================================================

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

// ==========================================
// 1. COSTANTI E STRUTTURE
// ==========================================
static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;



struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

struct ThreadPoolIterativeResult {
    IterativeResult result;
    ExecutionTimers timers;
};

// ==========================================
// 2. HELPER TIMER
// ==========================================

// ==========================================
// 3. OPERAZIONI VETTORIALI
// ==========================================
static double dot_parallel(const std::vector<double>& a, 
                           const std::vector<double>& b, 
                           ThreadPool& pool, 
                           std::size_t chunk_size) {
    const std::size_t n = a.size();
    std::vector<std::future<double>> futures;
    
    futures.reserve((n + chunk_size - 1) / chunk_size);

    for (std::size_t start = 0; start < n; start += chunk_size) {
        std::size_t end = std::min(start + chunk_size, n);

        futures.push_back(pool.enqueue([&a, &b, start, end]() {
            double partial_sum = 0.0;
            for (std::size_t i = start; i < end; ++i) {
                partial_sum += a[i] * b[i];
            }
            return partial_sum; 
        }));
    }

    double total_sum = 0.0;
    for (auto& fut : futures) {
        total_sum += fut.get();
    }
    return total_sum;
}

static double l2_norm_parallel(const std::vector<double>& x, 
                               ThreadPool& pool, 
                               std::size_t chunk_size) {
    return std::sqrt(dot_parallel(x, x, pool, chunk_size));
}

static void normalize_parallel(std::vector<double>& x, 
                               ThreadPool& pool, 
                               std::size_t chunk_size) {
    const double nrm = l2_norm_parallel(x, pool, chunk_size);
    const double inv = 1.0 / nrm;
    const std::size_t n = x.size();

    std::vector<std::future<void>> futures;
    
    futures.reserve((n + chunk_size - 1) / chunk_size);

    for (std::size_t start = 0; start < n; start += chunk_size) {
        std::size_t end = std::min(start + chunk_size, n);

        futures.push_back(pool.enqueue([&x, inv, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                x[i] *= inv;
            }
        }));
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

// ==========================================
// 4. SHIFT
// ==========================================
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// ==========================================
// 5. KERNEL SPMV
// ==========================================
static void spmv_csr_shifted_rows_parallel(const CSRMatrix& A,
                                           std::size_t row_shift,
                                           const std::vector<double>& x,
                                           std::vector<double>& y,
                                           ThreadPool& pool,
                                           std::size_t chunk_size) {
    const std::size_t n = A.n;
    std::vector<std::future<void>> futures;
    
    futures.reserve((n + chunk_size - 1) / chunk_size);

    for (std::size_t start = 0; start < n; start += chunk_size) {
        std::size_t end = std::min(start + chunk_size, n);

        futures.push_back(pool.enqueue([&A, row_shift, &x, &y, start, end, n]() {
            for (std::size_t i = start; i < end; ++i) {
                const std::size_t src_row = (i + n - row_shift) % n;

                double sum = 0.0;
                for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) {
                    sum += A.values[p] * x[A.col_idx[p]];
                }
                y[i] = sum;
            }
        }));
    }

    for (auto& fut : futures) {
        fut.get(); 
    }
}

// ==========================================
// 6. FUNZIONE ITERATIVA
// ==========================================
static ThreadPoolIterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                               std::uint64_t seed,
                                               ThreadPool& pool,
                                               std::size_t chunk_size,
                                               std::size_t norm_chunk_size,
                                               std::vector<double>* final_vector = nullptr) {
    ExecutionTimers timers;
    IterativeResult result;
    
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    std::vector<double> x(n);
    std::vector<double> y(n);

    const auto t_start_total = get_time_now();

    // Fase 1: Inizializzazione (RNG + normalizzazione)
    auto t0 = get_time_now();
    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }
    normalize_parallel(x, pool, norm_chunk_size); 
    timers.init_sec = get_elapsed_time(t0);

    std::size_t row_shift = 0;

    // Fase 2: Iterazioni
    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        if (iter > 0 && (iter % EPOCH_LEN) == 0) {
            t0 = get_time_now();
            row_shift = (row_shift + shift_rows) % n;
            timers.epoch_transition_sec += get_elapsed_time(t0);
        }

        t0 = get_time_now();
        spmv_csr_shifted_rows_parallel(A, row_shift, x, y, pool, chunk_size);
        timers.spmv_sec += get_elapsed_time(t0);

        t0 = get_time_now();
        normalize_parallel(y, pool, norm_chunk_size);
        timers.vector_ops_sec += get_elapsed_time(t0);

        x.swap(y);
    }

    // Fase 3: Diagnostica
    t0 = get_time_now();
    spmv_csr_shifted_rows_parallel(A, row_shift, x, y, pool, chunk_size);
    timers.spmv_sec += get_elapsed_time(t0);

    t0 = get_time_now();
    result.rayleigh = dot_parallel(x, y, pool, norm_chunk_size); 
    timers.vector_ops_sec += get_elapsed_time(t0);

    t0 = get_time_now();
    result.checksum = checksum_vector(x);
    timers.vector_ops_sec += get_elapsed_time(t0);
    result.final_row_shift = row_shift;

    if (final_vector != nullptr) {
        *final_vector = std::move(x);
    }

    timers.total_sec = get_elapsed_time(t_start_total);
    timers.computation_sec = timers.spmv_sec + timers.vector_ops_sec + timers.epoch_transition_sec;
    
    return ThreadPoolIterativeResult{result, timers};
}

// ==========================================
// 7. MAIN
// ==========================================
int main(int argc, char** argv) {
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t num_threads, chunk_size, norm_chunk_arg;
    std::string mode;
    std::string dump_vector_path;

    // Parametri base e di parallelismo sono obbligatori
    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "-t", num_threads) ||
        !read_arg_u64(argc, argv, "-c", chunk_size) ||
        !read_arg_u64(argc, argv, "-nc", norm_chunk_arg)) {
        
        usage(argv[0]);
        return 1;
    }

    // Lettura opzionale per seed e dump
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);
    
    const std::size_t norm_chunk_size = (norm_chunk_arg == 0) 
        ? ((n + num_threads - 1) / num_threads) 
        : static_cast<std::size_t>(norm_chunk_arg);

    std::cout << "SPARSE_ITERATION_CPP_THREADS\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";

    try {
        auto tg0 = get_time_now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const double generation_sec = get_elapsed_time(tg0);

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        ThreadPool pool(num_threads);

        ThreadPoolIterativeResult out = iterative_spmv_evolving(
            G.A, seed, pool, chunk_size, norm_chunk_size, final_vector_out);

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