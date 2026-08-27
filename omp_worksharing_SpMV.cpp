// ==============================================================================
// SPM "One-Shot" Project - OpenMP Work-Sharing Implementation (omp for)
// Regione parallela persistente: un solo fork/join per l'intera esecuzione.
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
// 1. COSTANTI E STRUTTURE (identiche alla versione a task)
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
// 2. SHIFT
// ==========================================
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// ==========================================
// 3. KERNEL PARALLELIZZATO (Work-Sharing)
// ==========================================
// IMPORTANTE: usa "#pragma omp for" (senza "parallel") perche' e' pensata
// per essere chiamata da TUTTI i thread di una regione "#pragma omp parallel"
// gia' aperta a monte. Non va mai chiamata da codice seriale o da dentro un
// blocco "single".
static void spmv_omp_for(const CSRMatrix& A,
                          std::size_t row_shift,
                          const std::vector<double>& x,
                          std::vector<double>& y,
                          std::size_t chunk_size) {
    const std::size_t n = A.n;

    // schedule(dynamic) e' la scelta giusta per SpMV: se alcune righe hanno
    // molti piu' non-zeri di altre (caso "irregular"), i thread che finiscono
    // prima richiedono subito il chunk successivo dalla coda centrale,
    // evitando che un thread resti sovraccarico mentre gli altri sono fermi.
    #pragma omp for schedule(dynamic, chunk_size)
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src_row = (i + n - row_shift) % n;

        double sum = 0.0;
        for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) {
            sum += A.values[p] * x[A.col_idx[p]];
        }
        y[i] = sum;
    }
}

// ==========================================
// 4. FUNZIONE ITERATIVA (Regione Parallela Persistente)
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

    // Variabili condivise per i timer e per la riduzione del dot product.
    // Devono vivere FUORI dalla regione parallela ed essere elencate come
    // "shared": tra due "single" consecutivi non e' garantito che sia lo
    // stesso thread ad eseguirli, quindi variabili "private" dichiarate
    // dentro la regione non sarebbero visibili in modo affidabile da un
    // blocco "single" all'altro.
    std::chrono::steady_clock::time_point t0;
    double dot_sum = 0.0;

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(A, x, y, n, shift_rows, chunk_size, norm_chunk_size, \
               row_shift, result, timers, seed, final_vector, t0, dot_sum)
    {
        // ---- Fase 1: Init (RNG + normalizzazione iniziale) ----
        #pragma omp single // "single" implica una barriera alla fine
        {
            t0 = get_time_now();
            SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
            for (double& v : x) {
                v = rng.next_unit();
            }
        } // tutti i thread aspettano qui finche' x non e' inizializzato

        #pragma omp single
        { dot_sum = 0.0; }

        #pragma omp for schedule(static, norm_chunk_size) reduction(+:dot_sum)
        for (std::size_t i = 0; i < n; ++i) {
            dot_sum += x[i] * x[i];
        } // barriera implicita del for

        #pragma omp single
        {
            const double inv = 1.0 / std::sqrt(dot_sum);
            for (std::size_t i = 0; i < n; ++i) {
                x[i] *= inv;
            }
            timers.init_sec = get_elapsed_time(t0);
        }

        // ---- Fase 2: iterazioni sulla matrice evolvente ----
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                #pragma omp single
                {
                    const auto t_ep0 = get_time_now();
                    row_shift = (row_shift + shift_rows) % n;
                    timers.epoch_transition_sec += get_elapsed_time(t_ep0);
                } // barriera implicita: tutti aspettano che row_shift sia aggiornato
            }

            // --- SpMV ---
            #pragma omp single
            { t0 = get_time_now(); } // tutti partono insieme all'istante t0 esatto

            spmv_omp_for(A, row_shift, x, y, chunk_size); // il for interno ha la sua barriera

            #pragma omp single
            { timers.spmv_sec += get_elapsed_time(t0); }

            // --- Normalizzazione ---
            #pragma omp single
            {
                t0 = get_time_now();
                dot_sum = 0.0;
            }

            #pragma omp for schedule(static, norm_chunk_size) reduction(+:dot_sum)
            for (std::size_t i = 0; i < n; ++i) {
                dot_sum += y[i] * y[i];
            }

            #pragma omp single
            {
                const double inv = 1.0 / std::sqrt(dot_sum);
                for (std::size_t i = 0; i < n; ++i) {
                    y[i] *= inv;
                }
                timers.vector_ops_sec += get_elapsed_time(t0);
                x.swap(y); // swap fatto in sicurezza, da un solo thread
            } // barriera implicita: nessuno inizia la nuova iterazione finche' lo swap non e' finito
        }

        // ---- Fase 3: diagnostica finale ----
        #pragma omp single
        { t0 = get_time_now(); }

        spmv_omp_for(A, row_shift, x, y, chunk_size);

        #pragma omp single
        {
            timers.spmv_sec += get_elapsed_time(t0);
            t0 = get_time_now();
            dot_sum = 0.0;
        }

        #pragma omp for schedule(static, norm_chunk_size) reduction(+:dot_sum)
        for (std::size_t i = 0; i < n; ++i) {
            dot_sum += x[i] * y[i];
        }

        #pragma omp single
        {
            result.rayleigh = dot_sum;
            timers.vector_ops_sec += get_elapsed_time(t0);

            t0 = get_time_now();
            result.checksum = checksum_vector(x);
            result.final_row_shift = row_shift;

            if (final_vector != nullptr) {
                *final_vector = std::move(x);
            }
            timers.vector_ops_sec += get_elapsed_time(t0);
        }
    } // fine regione parallela persistente

    timers.total_sec = get_elapsed_time(t_start_total);
    // Computation time is the sum of the components
    timers.computation_sec = timers.spmv_sec + timers.vector_ops_sec + timers.epoch_transition_sec;
    
    return OmpIterativeResult{result, timers};
}

// ==========================================
// 5. MAIN (identico alla versione a task)
// ==========================================
int main(int argc, char** argv) {
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t num_threads, chunk_size, norm_chunk_arg;
    std::string mode;
    std::string dump_vector_path;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "-t", num_threads) ||
        !read_arg_u64(argc, argv, "-c", chunk_size) ||
        !read_arg_u64(argc, argv, "-nc", norm_chunk_arg)) {

        usage(argv[0]); // Chiamata alla tua funzione in utils.hpp!
        return 1;
    }
    }

    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);

    const std::size_t norm_chunk_size = (norm_chunk_arg == 0)
        ? ((n + num_threads - 1) / num_threads)
        : static_cast<std::size_t>(norm_chunk_arg);

    std::cout << "SPARSE_ITERATION_OMP_WORKSHARING\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";

    try {
        const auto tg0 = get_time_now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const double generation_sec = get_elapsed_time(tg0);

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        const OmpIterativeResult out = iterative_spmv_evolving(
            G.A, seed, num_threads, chunk_size, norm_chunk_size, final_vector_out);

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