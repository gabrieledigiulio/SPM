// ==============================================================================
// SPM "One-Shot" Project - C++ Threads Implementation
//
// Istruzioni di compilazione:
//   g++ -O3 -std=c++20 -pthread -I . -Wall threadpool_SpMV.cpp -o threadpool_SpMV
//
// Istruzioni di esecuzione:
//   -n   Dimensione della matrice, NxN
//   -nz  Numero totale di elementi non nulli
//   -m   Modalità della matrice: regular o irregular
//   -t   Numero di thread da utilizzare (default: 4)
//   -c   Dimensione del chunk per SpMV (default: 1000)
//   -nc  Dimensione del chunk per operazioni vettoriali (0 = automatico, default: 0)
//   -s   Seed opzionale (default: 111)
//   --dump-vector FILE
//        File di output opzionale per il vettore normalizzato finale
//
// Esempio per test scaling e granularità:
//   ./threadpool_SpMV -n 500000 -nz 20000000 -m irregular -t 4 -c 1000 -nc 0
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


static constexpr std::uint32_t NUM_ITERS = 500; //[span_1](start_span)[span_1](end_span)
static constexpr std::uint32_t EPOCH_LEN = 25;  //[span_2](start_span)[span_2](end_span)


static double dot_parallel(const std::vector<double>& a, 
                           const std::vector<double>& b, 
                           ThreadPool& pool, 
                           std::size_t chunk_size) {
    const std::size_t n = a.size();
    std::vector<std::future<double>> futures;
    
    // Ottimizzazione: pre-allocazione della memoria per evitare riallocazioni
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
    
    // Ottimizzazione: pre-allocazione della memoria
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


static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17; //[span_3](start_span)[span_3](end_span)
    if ((s % 2) == 0) ++s;       //[span_4](start_span)[span_4](end_span)
    s %= n;                      //[span_5](start_span)[span_5](end_span)
    if (s == 0) s = 1;           //[span_6](start_span)[span_6](end_span)
    return s;                    //[span_7](start_span)[span_7](end_span)
}


static void spmv_csr_shifted_rows_parallel(const CSRMatrix& A,
                                           std::size_t row_shift,
                                           const std::vector<double>& x,
                                           std::vector<double>& y,
                                           ThreadPool& pool,
                                           std::size_t chunk_size) {
    const std::size_t n = A.n;
    std::vector<std::future<void>> futures;
    
    // Ottimizzazione: pre-allocazione della memoria
    futures.reserve((n + chunk_size - 1) / chunk_size);

    for (std::size_t start = 0; start < n; start += chunk_size) {
        std::size_t end = std::min(start + chunk_size, n);

        futures.push_back(pool.enqueue([&A, row_shift, &x, &y, start, end, n]() {
            for (std::size_t i = start; i < end; ++i) {
                const std::size_t src_row = (i + n - row_shift) % n; //[span_8](start_span)[span_8](end_span)

                double sum = 0.0;
                for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) { //[span_9](start_span)[span_9](end_span)
                    sum += A.values[p] * x[A.col_idx[p]]; //[span_10](start_span)[span_10](end_span)
                }
                y[i] = sum; //[span_11](start_span)[span_11](end_span)
            }
        }));
    }

    for (auto& fut : futures) {
        fut.get(); 
    }
}


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

static IterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                               std::uint64_t seed,
                                               ThreadPool& pool,
                                               std::size_t chunk_size,
                                               std::size_t norm_chunk_size,
                                               std::vector<double>* final_vector = nullptr) {
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n); //[span_12](start_span)[span_12](end_span)

    std::vector<double> x(n); //[span_13](start_span)[span_13](end_span)
    std::vector<double> y(n); //[span_14](start_span)[span_14](end_span)

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL); //[span_15](start_span)[span_15](end_span)
    for (double& v : x) {
        v = rng.next_unit(); //[span_16](start_span)[span_16](end_span)
    }
    
    normalize_parallel(x, pool, norm_chunk_size); 

    std::size_t row_shift = 0; //[span_17](start_span)[span_17](end_span)

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) { //[span_18](start_span)[span_18](end_span)
        if (iter > 0 && (iter % EPOCH_LEN) == 0) { //[span_19](start_span)[span_19](end_span)
            row_shift = (row_shift + shift_rows) % n; //[span_20](start_span)[span_20](end_span)
        }

        spmv_csr_shifted_rows_parallel(A, row_shift, x, y, pool, chunk_size);
        normalize_parallel(y, pool, norm_chunk_size);

        x.swap(y); //[span_21](start_span)[span_21](end_span)
    }

    spmv_csr_shifted_rows_parallel(A, row_shift, x, y, pool, chunk_size);
    const double rayleigh = dot_parallel(x, y, pool, norm_chunk_size); 
    const std::uint64_t checksum = checksum_vector(x); //[span_22](start_span)[span_22](end_span)

    if (final_vector != nullptr) {
        *final_vector = std::move(x); //[span_23](start_span)[span_23](end_span)
    }

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = row_shift
    };
}

int main(int argc, char** argv) {
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111; //[span_24](start_span)[span_24](end_span)
    std::uint64_t num_threads = 4;
    std::uint64_t chunk_size = 1000;
    std::uint64_t norm_chunk_arg = 0; 
    std::string mode;
    std::string dump_vector_path;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode)) { //[span_25](start_span)[span_25](end_span)
        std::cerr << "Uso: " << argv[0] << " -n N -nz K -m regular|irregular [-t THREADS] [-c CHUNK_SIZE] [-nc NORM_CHUNK_SIZE] [-s SEED] [--dump-vector FILE]\n";
        return 1;
    }

    (void)read_arg_u64(argc, argv, "-s", seed); //[span_26](start_span)[span_26](end_span)
    (void)read_arg_u64(argc, argv, "-t", num_threads);
    (void)read_arg_u64(argc, argv, "-c", chunk_size);
    (void)read_arg_u64(argc, argv, "-nc", norm_chunk_arg);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path); //[span_27](start_span)[span_27](end_span)

    const std::size_t n = static_cast<std::size_t>(n64); //[span_28](start_span)[span_28](end_span)
    
    const std::size_t norm_chunk_size = (norm_chunk_arg == 0) 
        ? ((n + num_threads - 1) / num_threads) 
        : static_cast<std::size_t>(norm_chunk_arg);

    std::cout << "SPARSE_ITERATION_CPP_THREADS\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";

    try {
        const auto tg0 = std::chrono::steady_clock::now(); //[span_29](start_span)[span_29](end_span)
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode); //[span_30](start_span)[span_30](end_span)
        const auto tg1 = std::chrono::steady_clock::now(); //[span_31](start_span)[span_31](end_span)

        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count(); //[span_32](start_span)[span_32](end_span)

        print_matrix_stats(G); //[span_33](start_span)[span_33](end_span)
        std::cout << "generation_time_sec=" << generation_sec << "\n\n"; //[span_34](start_span)[span_34](end_span)

        std::vector<double>  final_vector; //[span_35](start_span)[span_35](end_span)
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector; //[span_36](start_span)[span_36](end_span)

        ThreadPool pool(num_threads);

        const auto tc0 = std::chrono::steady_clock::now(); //[span_37](start_span)[span_37](end_span)
        const IterativeResult result = iterative_spmv_evolving(G.A, seed, pool, chunk_size, norm_chunk_size, final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now(); //[span_38](start_span)[span_38](end_span)

        const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count(); //[span_39](start_span)[span_39](end_span)

        std::cout << std::setprecision(15); //[span_40](start_span)[span_40](end_span)
        std::cout << "rayleigh=" << result.rayleigh << "\n"; //[span_41](start_span)[span_41](end_span)
        std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n"; //[span_42](start_span)[span_42](end_span)

        std::cout << std::fixed << std::setprecision(6); //[span_43](start_span)[span_43](end_span)
        std::cout << "Time (sec) = " << computation_sec << "\n"; //[span_44](start_span)[span_44](end_span)

        if (!dump_vector_path.empty()) { //[span_45](start_span)[span_45](end_span)
            dump_vector(dump_vector_path, final_vector); //[span_46](start_span)[span_46](end_span)
            std::cout << "vector_dump=" << dump_vector_path << "\n"; //[span_47](start_span)[span_47](end_span)
        }
    } catch (const std::exception& e) { //[span_48](start_span)[span_48](end_span)
        std::cerr << "Error: " << e.what() << "\n"; //[span_49](start_span)[span_49](end_span)
        return 1; //[span_50](start_span)[span_50](end_span)
    }

    return 0; //[span_51](start_span)[span_51](end_span)
}
