// ==============================================================================
// SPM "One-Shot" Project - OpenMP Task-Based Implementation
//
// Istruzioni di compilazione:
//   g++ -O3 -std=c++20 -fopenmp -I . -Wall omp_tasks_SpMV.cpp -o omp_tasks_SpMV
//
// Istruzioni di esecuzione:
//   -n   Dimensione della matrice, NxN
//   -nz  Numero totale di elementi non nulli
//   -m   Modalità della matrice: regular o irregular
//   -t   Numero di thread OpenMP da utilizzare (default: 4)
//   -c   Dimensione del chunk per SpMV (default: 1000)
//   -nc  Dimensione del chunk per operazioni vettoriali (0 = automatico, default: 0)
//   -s   Seed opzionale (default: 111)
//   --dump-vector FILE
//        File di output opzionale per il vettore normalizzato finale
//
// Esempio per test scaling e granularità:
//   ./omp_tasks_SpMV -n 500000 -nz 20000000 -m irregular -t 4 -c 1000 -nc 0
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
// 1. COSTANTI
// ==========================================
static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;

// ==========================================
// 2. OPERAZIONI VETTORIALI (Parallelizzate)
// ==========================================

// Nota: nessuna regione "parallel" propria -- va chiamata da dentro un
// contesto parallelo esistente (qui, dal blocco "single" della regione
// persistente). Il taskgroup interno e' pero' un punto di sincronizzazione
// locale: la funzione ritorna solo quando tutti i task di riduzione sono
// completati, quindi il chiamante NON deve fare un taskwait separato.
static double dot_omp_tasks(const std::vector<double>& a,
                            const std::vector<double>& b,
                            std::size_t chunk_size) {
    const std::size_t n = a.size();
    double sum = 0.0;

    #pragma omp taskgroup task_reduction(+:sum)
    {
        for (std::size_t start = 0; start < n; start += chunk_size) {
            const std::size_t end = std::min(start + chunk_size, n);

            // firstprivate(start, end): range congelato per questo task.
            // shared(a, b): dati in sola lettura, nessuna scrittura concorrente.
            // in_reduction(+:sum): partecipa alla riduzione del taskgroup esterno.
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
    } // fine taskgroup: "sum" contiene il totale, tutti i task sono finiti

    return sum;
}

static double l2_norm_omp_tasks(const std::vector<double>& x,
                                std::size_t chunk_size) {
    return std::sqrt(dot_omp_tasks(x, x, chunk_size));
}

// Nota: nessuna riduzione qui -- ogni task scrive solo la propria fetta
// di x, quindi un taskgroup "semplice" basta per aspettare la scrittura.
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
    } // fine taskgroup: tutte le scritture su x sono completate
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

// Nota: come le funzioni vettoriali, non apre una propria regione
// "parallel" -- viene chiamata da dentro il blocco "single" della regione
// persistente. Crea solo i task; è compito del chiamante fare taskwait
// dopo, dato che qui NON c'è un taskgroup che faccia da barriera locale.
static void spmv_omp_tasks(const CSRMatrix& A,
                           std::size_t row_shift,
                           const std::vector<double>& x,
                           std::vector<double>& y,
                           std::size_t chunk_size) {
    const std::size_t n = A.n;

    for (std::size_t start = 0; start < n; start += chunk_size) {
        const std::size_t end = std::min(start + chunk_size, n);

        // firstprivate(start, end, row_shift, n): valori congelati per il task.
        // shared(A, x, y): dati pesanti; ogni task scrive solo y[i] per i
        // nel proprio range, quindi nessuna sovrapposizione in scrittura.
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
// 5. FUNZIONE ITERATIVA E MAIN
// ==========================================
struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

static IterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                               std::uint64_t seed,
                                               std::size_t num_threads,
                                               std::size_t chunk_size,
                                               std::size_t norm_chunk_size,
                                               std::vector<double>* final_vector = nullptr) {
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    std::vector<double> x(n);
    std::vector<double> y(n);

    // Fase 1: inizializzazione -- sequenziale, come nel riferimento.
    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }

    std::size_t row_shift = 0;
    double rayleigh = 0.0;
    std::uint64_t checksum = 0;

    // Regione parallela persistente: aperta una sola volta per tutta la
    // computazione (normalizzazione iniziale + 500 iterazioni + fase finale).
    // Un solo thread ("single") fa da produttore di task; gli altri thread
    // del team eseguono i task man mano che vengono creati.
    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(A, x, y, n, shift_rows, chunk_size, norm_chunk_size, \
               row_shift, rayleigh, checksum, final_vector)
    {
        #pragma omp single
        {
            // Fase 1 (continua): normalizzazione iniziale di x.
            // normalize_omp_tasks fa gia' da barriera internamente (taskgroup).
            normalize_omp_tasks(x, norm_chunk_size);

            // Fase 2: iterazioni sulla matrice evolvente.
            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    row_shift = (row_shift + shift_rows) % n;
                }

                // --- Fase SpMV: crea i task, poi aspetta che finiscano ---
                spmv_omp_tasks(A, row_shift, x, y, chunk_size);
                #pragma omp taskwait
                // Barriera logica: da qui in poi e' garantito che tutti i
                // task SpMV abbiano finito di scrivere y.

                // --- Fase normalizzazione: barriera gia' inclusa (taskgroup) ---
                normalize_omp_tasks(y, norm_chunk_size);

                // Sicuro: nessun task sta leggendo/scrivendo x o y adesso.
                x.swap(y);
            }

            // Fase 3: diagnostica finale.
            spmv_omp_tasks(A, row_shift, x, y, chunk_size);
            #pragma omp taskwait

            rayleigh = dot_omp_tasks(x, y, norm_chunk_size);
            checksum = checksum_vector(x);

            if (final_vector != nullptr) {
                *final_vector = std::move(x);
            }
        } // fine single (barrier implicita in uscita)
    } // fine parallel

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = row_shift
    };
}

int main(int argc, char** argv) {
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t num_threads = 4;
    std::uint64_t chunk_size = 1000;
    std::uint64_t norm_chunk_arg = 0;
    std::string mode;
    std::string dump_vector_path;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode)) {
        std::cerr << "Uso: " << argv[0] << " -n N -nz K -m regular|irregular [-t THREADS] [-c CHUNK_SIZE] [-nc NORM_CHUNK_SIZE] [-s SEED] [--dump-vector FILE]\n";
        return 1;
    }

    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_u64(argc, argv, "-t", num_threads);
    (void)read_arg_u64(argc, argv, "-c", chunk_size);
    (void)read_arg_u64(argc, argv, "-nc", norm_chunk_arg);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);

    const std::size_t norm_chunk_size = (norm_chunk_arg == 0)
        ? ((n + num_threads - 1) / num_threads)
        : static_cast<std::size_t>(norm_chunk_arg);

    std::cout << "SPARSE_ITERATION_OMP_TASKS\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";

    try {
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();

        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result = iterative_spmv_evolving(
            G.A, seed, num_threads, chunk_size, norm_chunk_size, final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now();

        const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << computation_sec << "\n";

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