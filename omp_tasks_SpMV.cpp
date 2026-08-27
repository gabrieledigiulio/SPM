// ==============================================================================
// SPM "One-Shot" Project - OpenMP Task-Based Implementation
// ==============================================================================

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

struct OmpIterativeResult {
    IterativeResult result;
    ExecutionTimers timers;
};

// ==========================================
// 2. OPERAZIONI VETTORIALI (Parallelizzate)
// ==========================================

static double dot_omp_tasks(const std::vector<double>& a,
                            const std::vector<double>& b,
                            std::size_t chunk_size) {
    const std::size_t n = a.size();
    double sum = 0.0;

    #pragma omp taskgroup task_reduction(+:sum)
    {
        for (std::size_t start = 0; start < n; start += chunk_size) {
            const std::size_t end = std::min(start + chunk_size, n);

            #pragma omp task firstprivate(start, end) shared(a, b) \
                              in_reduction(+:sum) default(none)
            {
                double partial = 0.0;
                for (std::size_t i = start; i < end; ++i) {
                    partial += a[i] * b[i];
                }
                sum += partial;
            }
        }
    } 
    return sum;
}

static double l2_norm_omp_tasks(const std::vector<double>& x,
                                std::size_t chunk_size) {
    return std::sqrt(dot_omp_tasks(x, x, chunk_size));
}

static void normalize_omp_tasks(std::vector<double>& x,
                                std::size_t chunk_size) {
    const double nrm = l2_norm_omp_tasks(x, chunk_size);
    const double inv = 1.0 / nrm;
    const std::size_t n = x.size();

    #pragma omp taskgroup
    {
        for (std::size_t start = 0; start < n; start += chunk_size) {
            const std::size_t end = std::min(start + chunk_size, n);

            #pragma omp task firstprivate(start, end, inv) shared(x) \
                              default(none)
            {
                for (std::size_t i = start; i < end; ++i) {
                    x[i] *= inv;
                }
            }
        }
    } 
}

// ==========================================
// 3. SHIFT
// ==========================================
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// ==========================================
// 4. KERNEL PARALLELIZZATO
// ==========================================

static void spmv_omp_tasks(const CSRMatrix& A,
                           std::size_t row_shift,
                           const std::vector<double>& x,
                           std::vector<double>& y,
                           std::size_t chunk_size) {
    const std::size_t n = A.n;

    for (std::size_t start = 0; start < n; start += chunk_size) {
        const std::size_t end = std::min(start + chunk_size, n);

        #pragma omp task firstprivate(start, end, row_shift, n) \
                          shared(A, x, y) default(none)
        {
            for (std::size_t i = start; i < end; ++i) {
                const std::size_t src_row = (i + n - row_shift) % n;

                double sum = 0.0;
                for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) {
                    sum += A.values[p] * x[A.col_idx[p]];
                }
                y[i] = sum;
            }
        }
    }
}

// ==========================================
// 5. FUNZIONE ITERATIVA 
// ==========================================

static OmpIterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                               std::uint64_t seed,
                                               std::size_t num_threads,
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

    std::size_t row_shift = 0;

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(A, x, y, n, shift_rows, chunk_size, norm_chunk_size, \
               row_shift, result, final_vector, timers, seed)
    {
        #pragma omp single
        {
            // Fase 1: Init (RNG + normalizzazione iniziale)
            auto t_init0 = get_time_now();
            SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
            for (double& v : x) {
                v = rng.next_unit();
            }
            normalize_omp_tasks(x, norm_chunk_size);
            timers.init_sec = get_elapsed_time(t_init0);

            // Fase 2: iterazioni sulla matrice evolvente.
            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    auto t_ep0 = get_time_now();
                    row_shift = (row_shift + shift_rows) % n;
                    timers.epoch_transition_sec += get_elapsed_time(t_ep0);
                }

                // SpMV
                auto t_spmv0 = get_time_now();
                spmv_omp_tasks(A, row_shift, x, y, chunk_size);
                #pragma omp taskwait
                timers.spmv_sec += get_elapsed_time(t_spmv0);

                // Normalizzazione
                auto t_vec0 = get_time_now();
                normalize_omp_tasks(y, norm_chunk_size);
                timers.vector_ops_sec += get_elapsed_time(t_vec0);

                x.swap(y);
            }

            // Fase 3: diagnostica finale.
            auto t_spmv_final = get_time_now();
            spmv_omp_tasks(A, row_shift, x, y, chunk_size);
            #pragma omp taskwait
            timers.spmv_sec += get_elapsed_time(t_spmv_final);

            auto t_vec_final = get_time_now();
            result.rayleigh = dot_omp_tasks(x, y, norm_chunk_size);
            timers.vector_ops_sec += get_elapsed_time(t_vec_final);
            
            auto t_chk0 = get_time_now();
            result.checksum = checksum_vector(x);
            timers.vector_ops_sec += get_elapsed_time(t_chk0);
            result.final_row_shift = row_shift;

            if (final_vector != nullptr) {
                *final_vector = std::move(x);
            }
        } 
    } 

    timers.total_sec = get_elapsed_time(t_start_total);
    return OmpIterativeResult{result, timers};
}

// ==========================================
// 6. MAIN
// ==========================================

int main(int argc, char** argv) {
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t num_threads, chunk_size, norm_chunk_arg;
    std::string mode;
    std::string dump_vector_path;

    // -n, -nz, -m, -t, -c, -nc sono TUTTI obbligatori
    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "-t", num_threads) ||
        !read_arg_u64(argc, argv, "-c", chunk_size) ||
        !read_arg_u64(argc, argv, "-nc", norm_chunk_arg)) {
        
        usage(argv[0]);
        return 1;
    }

    // Solo seed e dump-vector rimangono opzionali
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);

    const std::size_t norm_chunk_size = (norm_chunk_arg == 0)
        ? ((n + num_threads - 1) / num_threads)
        : static_cast<std::size_t>(norm_chunk_arg);

    std::cout << "SPARSE_ITERATION_OMP_TASKS\n";
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

        OmpIterativeResult out = iterative_spmv_evolving(
            G.A, seed, num_threads, chunk_size, norm_chunk_size, final_vector_out);

        out.timers.computation_sec = out.timers.total_sec; // Compute sec = total sec here
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