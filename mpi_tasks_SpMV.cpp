// ==============================================================================
// SPM "One-Shot" Project - MPI+OpenMP Implementation
//
// Design (see report for full justification):
//   - Physical row partitioning, balanced by NNZ, computed once at setup and
//     NEVER changed (no physical redistribution at epoch boundaries).
//   - Sparsity locality analysis (analyze_sparsity.cpp) showed that a
//     ghost-cell / Inspector-Executor design brings negligible communication
//     savings for this matrix (each physical block needs ~92% of the columns
//     of x), so we use a full MPI_Allgatherv of the vector each iteration
//     instead: fixed counts/displacements, computed once, never recomputed.
//   - The row_shift is handled ENTIRELY as a local, in-memory permutation:
//     a physical row p always contributes its result to logical index
//     (p + row_shift) mod n. Since every rank ends up with the complete
//     vector after the Allgatherv, this permutation requires zero network
//     traffic -- it's a local reordering, parallelized with OpenMP.
//
// Istruzioni di compilazione:
//   mpic++ -O3 -std=c++20 -fopenmp -I . -Wall mpi_omp_SpMV.cpp -o mpi_omp_SpMV
//
// Istruzioni di esecuzione:
//   -n   Dimensione della matrice, NxN
//   -nz  Numero totale di elementi non nulli
//   -m   Modalità della matrice: regular o irregular
//   -c   Dimensione del chunk per SpMV locale (default: 1000)
//   -nc  Dimensione del chunk per operazioni vettoriali locali (0 = automatico)
//   -s   Seed opzionale (default: 111)
//   --dump-vector FILE
//        File di output opzionale per il vettore normalizzato finale
//   (i thread OpenMP per processo si controllano con OMP_NUM_THREADS)
//
// Esempio (4 nodi, 4 processi, 4 thread ciascuno):
//   mpirun -np 4 -x OMP_NUM_THREADS=4 ./mpi_omp_SpMV -n 500000 -nz 20000000 -m irregular
// ==============================================================================

#include "matrix_generation.hpp"
#include "utils.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ==========================================
// 1. COSTANTI
// ==========================================
static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;

// ==========================================
// 2. PARTIZIONAMENTO FISICO E STRUTTURE DISTRIBUITE
// ==========================================

// Range di righe FISICHE (indici nella CSR originale) assegnate a un rank.
// Fisso per tutta l'esecuzione: non viene mai ricalcolato alle epoche.
struct BlockRange {
    std::size_t row_begin = 0;
    std::size_t row_end   = 0; // esclusivo
};

// Sottomatrice locale di un rank: stesse colonne globali della matrice
// originale (indicizzano ancora nel vettore x completo), ma row_ptr
// ribasato a partire da 0 per le sole righe possedute localmente.
struct LocalMatrix {
    std::size_t row_begin = 0; // offset fisico globale della prima riga locale
    std::size_t num_rows  = 0;
    std::vector<std::uint64_t> row_ptr; // size num_rows + 1, row_ptr[0] == 0
    std::vector<std::uint32_t> col_idx; // indici di colonna GLOBALI (0..n-1)
    std::vector<double> values;
};

// Stessa logica di analyze_sparsity.cpp: taglia le righe fisiche in
// num_blocks blocchi contigui bilanciati per NNZ (non per numero di righe).
static std::vector<BlockRange> partition_rows_by_nnz(const CSRMatrix& A,
                                                      std::size_t num_blocks) {
    const std::size_t n = A.n;
    const std::uint64_t total_nnz = A.row_ptr[n];
    const std::uint64_t target_per_block = (total_nnz + num_blocks - 1) / num_blocks;

    std::vector<BlockRange> blocks;
    blocks.reserve(num_blocks);

    std::size_t row_start = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (row_start >= n) {
            blocks.push_back({row_start, row_start});
            continue;
        }

        const std::uint64_t nnz_floor = A.row_ptr[row_start];
        const std::uint64_t target_prefix = nnz_floor + target_per_block;

        std::size_t row_end;
        if (b == num_blocks - 1) {
            row_end = n; // l'ultimo blocco prende tutto il resto
        } else {
            row_end = std::distance(
                A.row_ptr.begin(),
                std::lower_bound(A.row_ptr.begin() + row_start, A.row_ptr.end(), target_prefix));
            row_end = std::clamp(row_end, row_start + 1, n);
        }

        blocks.push_back({row_start, row_end});
        row_start = row_end;
    }

    return blocks;
}

// Estrae la sottomatrice locale di un blocco dalla CSR globale (usata dal
// rank 0 per costruire il pezzo da inviare -- o da tenere per se stesso).
static LocalMatrix extract_local_matrix(const CSRMatrix& A, const BlockRange& range) {
    LocalMatrix L;
    L.row_begin = range.row_begin;
    L.num_rows  = range.row_end - range.row_begin;

    const std::uint64_t begin_p = A.row_ptr[range.row_begin];
    const std::uint64_t end_p   = A.row_ptr[range.row_end];

    L.row_ptr.resize(L.num_rows + 1);
    for (std::size_t i = 0; i <= L.num_rows; ++i) {
        L.row_ptr[i] = A.row_ptr[range.row_begin + i] - begin_p;
    }

    L.col_idx.assign(A.col_idx.begin() + begin_p, A.col_idx.begin() + end_p);
    L.values.assign(A.values.begin() + begin_p, A.values.begin() + end_p);

    return L;
}

// Invia una LocalMatrix gia' estratta al rank di destinazione (usata da
// rank 0 durante il setup). Comunicazione punto-a-punto semplice: la
// distribuzione avviene una sola volta, fuori dalla regione temporizzata
// dell'iterazione, quindi non serve essere sofisticati qui.
static void send_local_matrix(const LocalMatrix& L, int dest_rank) {
    std::uint64_t sizes[3] = {L.num_rows, L.col_idx.size(), L.row_begin};
    MPI_Send(sizes, 3, MPI_UINT64_T, dest_rank, 0, MPI_COMM_WORLD);

    MPI_Send(L.row_ptr.data(), static_cast<int>(L.row_ptr.size()),
              MPI_UINT64_T, dest_rank, 1, MPI_COMM_WORLD);
    MPI_Send(L.col_idx.data(), static_cast<int>(L.col_idx.size()),
              MPI_UINT32_T, dest_rank, 2, MPI_COMM_WORLD);
    MPI_Send(L.values.data(), static_cast<int>(L.values.size()),
              MPI_DOUBLE, dest_rank, 3, MPI_COMM_WORLD);
}

// Riceve la propria LocalMatrix dal rank 0.
static LocalMatrix recv_local_matrix(int source_rank) {
    LocalMatrix L;

    std::uint64_t sizes[3];
    MPI_Recv(sizes, 3, MPI_UINT64_T, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    L.num_rows  = sizes[0];
    const std::uint64_t nnz_local = sizes[1];
    L.row_begin = sizes[2];

    L.row_ptr.resize(L.num_rows + 1);
    L.col_idx.resize(nnz_local);
    L.values.resize(nnz_local);

    MPI_Recv(L.row_ptr.data(), static_cast<int>(L.row_ptr.size()),
             MPI_UINT64_T, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(L.col_idx.data(), static_cast<int>(L.col_idx.size()),
             MPI_UINT32_T, source_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(L.values.data(), static_cast<int>(L.values.size()),
             MPI_DOUBLE, source_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return L;
}

// Fase di setup: il rank 0 genera l'intera matrice, calcola il
// partizionamento fisico bilanciato per NNZ, e distribuisce ad ogni rank
// la propria sottomatrice locale. Ritorna anche "all_blocks" (uguale su
// tutti i rank) perche' serve per calcolare i counts/displs fissi
// dell'Allgatherv usato poi nel ciclo iterativo.
//
// generation_sec e distribution_sec sono tempi diagnostici, ESCLUSI dalla
// regione temporizzata del calcolo iterativo (come nel sequenziale, dove
// la generazione non e' parte del "computation time").
//
// Nota di robustezza: se generate_matrix lancia un'eccezione sul rank 0,
// gli altri rank sarebbero bloccati per sempre sulla MPI_Bcast successiva
// (non hanno modo di saperlo). Per questo il rank 0 trasmette prima un
// flag di esito ("ok"): se e' 0, TUTTI i rank lanciano un'eccezione
// coerente e main() puo' fare MPI_Abort in modo pulito su tutto il job.
static LocalMatrix setup_and_distribute(std::size_t n, std::uint64_t nz,
                                        std::uint64_t seed, const std::string& mode,
                                        int rank, int num_ranks,
                                        std::vector<BlockRange>& all_blocks,
                                        double& generation_sec,
                                        double& distribution_sec) {
    generation_sec = 0.0;
    distribution_sec = 0.0;

    if (rank == 0) {
        int ok = 1;
        std::string error_msg;
        GeneratedMatrix G;

        const auto tg0 = std::chrono::steady_clock::now();
        try {
            G = generate_matrix(n, nz, seed, mode);
        } catch (const std::exception& e) {
            ok = 0;
            error_msg = e.what();
        }
        const auto tg1 = std::chrono::steady_clock::now();
        generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) {
            throw std::runtime_error("generazione matrice fallita: " + error_msg);
        }

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n";

        all_blocks = partition_rows_by_nnz(G.A, static_cast<std::size_t>(num_ranks));

        // Bcast: tutti i rank devono conoscere il partizionamento completo
        // per calcolare autonomamente i counts/displs fissi dell'Allgatherv
        // usato nel ciclo iterativo (root = rank 0, invio in un colpo solo,
        // piu' idiomatico e sicuro del loop manuale di Send precedente).
        MPI_Bcast(all_blocks.data(), num_ranks * 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        const auto td0 = std::chrono::steady_clock::now();

        LocalMatrix local;
        for (int r = 0; r < num_ranks; ++r) {
            LocalMatrix piece = extract_local_matrix(G.A, all_blocks[r]);
            if (r == 0) {
                local = std::move(piece);
            } else {
                send_local_matrix(piece, r);
            }
        }

        const auto td1 = std::chrono::steady_clock::now();
        distribution_sec = std::chrono::duration<double>(td1 - td0).count();

        return local;
    } else {
        int ok = 1;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) {
            throw std::runtime_error("generazione matrice fallita sul rank 0");
        }

        all_blocks.resize(num_ranks);
        MPI_Bcast(all_blocks.data(), num_ranks * 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        return recv_local_matrix(0);
    }
}

// ==========================================
// 3. SHIFT
// ==========================================
// Identica al sequenziale: e' pura aritmetica su n, nessuna dipendenza
// dal partizionamento distribuito.
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// ==========================================
// 4. OPERAZIONI VETTORIALI LOCALI (OpenMP task-based)
// ==========================================
//
// A differenza di omp_tasks_SpMV.cpp, qui le funzioni operano su una
// SOTTOSEZIONE [begin, begin+count) di un vettore, non sull'intero
// vettore: ogni rank possiede fisicamente solo la propria porzione
// (dimensione = numero di righe fisiche locali). La combinazione tra
// le porzioni dei vari rank avviene fuori da queste funzioni, tramite
// MPI_Allreduce sul valore scalare gia' ridotto localmente.
//
// Nessuna di queste funzioni apre una propria regione "parallel": vanno
// chiamate da dentro il blocco "single" della regione OpenMP persistente
// (stessa convenzione di omp_tasks_SpMV.cpp).

// Riduzione locale (dot product) su una fetta [begin, begin+count) di due
// vettori. Usata sia per il prodotto scalare vero e proprio (rayleigh
// finale) sia per la somma dei quadrati (norma) con a == b.
static double local_dot_omp_tasks(const std::vector<double>& a,
                                  const std::vector<double>& b,
                                  std::size_t begin, std::size_t count,
                                  std::size_t chunk_size) {
    double sum = 0.0;

    #pragma omp taskgroup task_reduction(+:sum)
    {
        for (std::size_t off = 0; off < count; off += chunk_size) {
            const std::size_t start = begin + off;
            const std::size_t end   = begin + std::min(off + chunk_size, count);

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
    } // fine taskgroup: "sum" e' il totale LOCALE (solo su questa fetta)

    return sum;
}

// Scala in-place una fetta [begin, begin+count) di v per "factor". Nessuna
// riduzione: ogni task scrive solo la propria porzione, un taskgroup
// "semplice" basta per aspettare che tutte le scritture siano completate.
static void local_scale_omp_tasks(std::vector<double>& v,
                                  std::size_t begin, std::size_t count,
                                  double factor, std::size_t chunk_size) {
    #pragma omp taskgroup
    {
        for (std::size_t off = 0; off < count; off += chunk_size) {
            const std::size_t start = begin + off;
            const std::size_t end   = begin + std::min(off + chunk_size, count);

            #pragma omp task firstprivate(start, end, factor) shared(v) \
                              default(none)
            {
                for (std::size_t i = start; i < end; ++i) {
                    v[i] *= factor;
                }
            }
        }
    } // fine taskgroup: tutte le scritture su v[begin, begin+count) completate
}

// ==========================================
// 5. KERNEL SPMV LOCALE + ALLGATHERV + PERMUTAZIONE DELLO SHIFT
// ==========================================

// Kernel SpMV locale: NON dipende da row_shift. Il calcolo di una riga
// fisica usa solo i propri nonzero (fissi per sempre) e il vettore x
// completo (gia' disponibile su ogni rank dall'iterazione precedente).
// row_shift entra in gioco SOLO dopo, nella permutazione che decide dove
// va scritto il risultato -- qui produciamo solo y_phys, indicizzato per
// riga FISICA locale (0..num_rows-1), non per indice logico.
//
// Nessuna regione "parallel" propria: chiamata da dentro il blocco
// "single" della regione persistente. Il chiamante deve fare taskwait
// dopo, dato che qui non c'e' un taskgroup che faccia da barriera locale
// (stessa convenzione di spmv_omp_tasks in omp_tasks_SpMV.cpp).
static void spmv_local_omp_tasks(const LocalMatrix& L,
                                 const std::vector<double>& x_full,
                                 std::vector<double>& y_phys,
                                 std::size_t chunk_size) {
    for (std::size_t start = 0; start < L.num_rows; start += chunk_size) {
        const std::size_t end = std::min(start + chunk_size, L.num_rows);

        // firstprivate(start, end): range locale congelato per il task.
        // shared(L, x_full, y_phys): ogni task scrive solo y_phys[i] per
        // i nel proprio range, mai sovrapposto tra task.
        #pragma omp task firstprivate(start, end) shared(L, x_full, y_phys) \
                          default(none)
        {
            for (std::size_t i = start; i < end; ++i) {
                double sum = 0.0;
                for (std::uint64_t p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p) {
                    sum += L.values[p] * x_full[L.col_idx[p]];
                }
                y_phys[i] = sum;
            }
        }
    }
}

// Piano fisso per MPI_Allgatherv: counts/displs indicizzati per rank,
// derivati UNA SOLA VOLTA dal partizionamento fisico (all_blocks), che
// non cambia mai durante l'esecuzione. Nessuna epoca lo ricalcola.
struct AllgatherPlan {
    std::vector<int> counts; // counts[r] = numero di righe fisiche del rank r
    std::vector<int> displs; // displs[r] = offset fisico (== row_begin) del rank r
};

static AllgatherPlan build_allgather_plan(const std::vector<BlockRange>& all_blocks) {
    AllgatherPlan plan;
    plan.counts.reserve(all_blocks.size());
    plan.displs.reserve(all_blocks.size());

    for (const BlockRange& b : all_blocks) {
        // static_cast a int: sicuro per le taglie di matrice di questo
        // progetto (n dell'ordine di 10^5-10^6, ben dentro INT_MAX).
        plan.counts.push_back(static_cast<int>(b.row_end - b.row_begin));
        plan.displs.push_back(static_cast<int>(b.row_begin));
    }

    return plan;
}

// Raccoglie il vettore fisico completo (indicizzato per riga FISICA,
// 0..n-1) da tutti i rank. Bloccante: e' la comunicazione collettiva
// misurata come "communication time" nel breakdown richiesto dal report.
static void gather_full_vector(const std::vector<double>& y_phys_local,
                               std::vector<double>& y_phys_full,
                               const AllgatherPlan& plan) {
    MPI_Allgatherv(y_phys_local.data(), static_cast<int>(y_phys_local.size()), MPI_DOUBLE,
                   y_phys_full.data(), plan.counts.data(), plan.displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
}

// Permutazione locale: y_phys_full e' indicizzato per riga FISICA p
// (0..n-1); il suo contenuto appartiene alla posizione LOGICA
// (p + row_shift) mod n. Questa e' l'unica "evoluzione della matrice"
// che serve applicare, ed e' puro riordino in RAM -- zero comunicazione,
// eseguito ridondantemente (ma in parallelo) su ogni rank, perche' ogni
// rank ha gia' l'intero y_phys_full disponibile dopo la Allgatherv.
//
// Nessuna riduzione, nessuna dipendenza tra scritture: la mappa
// p -> (p + row_shift) mod n e' una biiezione, quindi task diversi
// scrivono sempre indici di destinazione disgiunti in x_next.
static void apply_shift_permutation_omp_tasks(const std::vector<double>& y_phys_full,
                                              std::vector<double>& x_next,
                                              std::size_t row_shift,
                                              std::size_t n,
                                              std::size_t chunk_size) {
    #pragma omp taskgroup
    {
        for (std::size_t start = 0; start < n; start += chunk_size) {
            const std::size_t end = std::min(start + chunk_size, n);

            #pragma omp task firstprivate(start, end, row_shift, n) \
                              shared(y_phys_full, x_next) default(none)
            {
                for (std::size_t p = start; p < end; ++p) {
                    const std::size_t dest = (p + row_shift) % n;
                    x_next[dest] = y_phys_full[p];
                }
            }
        }
    } // fine taskgroup: tutte le scritture su x_next sono completate
}

// ==========================================
// 6. CICLO ITERATIVO IBRIDO (MPI + OpenMP)
// ==========================================

// Breakdown dei tempi richiesto dal report. Tutti i campi sono espressi in
// secondi e vengono accumulati (+=) iterazione per iterazione dentro il
// thread "single" che guida il loop -- nessuna sincronizzazione necessaria
// per scriverli, dato che un solo thread li tocca.
struct ExecutionTimers {
    double init_sec              = 0.0; // generazione + normalizzazione iniziale di x (fuori loop, non "per iterazione")
    double local_compute_sec     = 0.0; // spmv locale + dot locale + scale locale + permutazione (mai comunicazione)
    double reduction_sec         = 0.0; // MPI_Allreduce (norma, scalare 8 byte)
    double communication_sec     = 0.0; // MPI_Allgatherv (vettore, n*8 byte)
    double epoch_transition_sec  = 0.0; // aggiornamento di row_shift (atteso ~0, lo misuriamo per dimostrarlo)
    double total_sec             = 0.0; // l'intera iterative_spmv_evolving_mpi_omp, init+loop+fase finale
};

struct MpiIterativeResult {
    IterativeResult result;   // rayleigh, checksum, final_row_shift -- validi SOLO su rank 0
    ExecutionTimers timers;   // validi su OGNI rank (per l'analisi di imbalance nel report)
};

// L               : sottomatrice fisica locale di questo rank (mai modificata)
// n                : dimensione globale del problema
// plan             : counts/displs fissi per l'Allgatherv, calcolati una volta in main()
// chunk_size       : granularita' dei task SpMV (righe fisiche locali)
// norm_chunk_size  : granularita' dei task per dot/scale/permutazione
// final_vector_out : se non nullptr, riceve x finale COMPLETO -- va popolato solo su rank 0
//                    (e' l'unico rank che scrive il dump, coerente col resto del progetto)
static MpiIterativeResult
iterative_spmv_evolving_mpi_omp(const LocalMatrix& L, std::size_t n,
                                std::uint64_t seed,
                                const AllgatherPlan& plan,
                                int rank,
                                std::size_t chunk_size,
                                std::size_t norm_chunk_size,
                                std::vector<double>* final_vector_out) {
    ExecutionTimers timers;
    IterativeResult result;

    const std::size_t shift_rows = compute_shift_rows(n);
    std::size_t row_shift = 0;

    // Buffer allocati UNA VOLTA, riusati per tutte le 500 iterazioni (mai
    // resize dentro il loop: sarebbe rumore di allocazione nel timing).
    std::vector<double> x_full(n);              // vettore logico completo, replicato su ogni rank
    std::vector<double> x_next(n);               // buffer di destinazione della permutazione
    std::vector<double> y_phys_local(L.num_rows); // output SpMV locale, indicizzato per riga FISICA
    std::vector<double> y_phys_full(n);           // dopo l'Allgatherv, indicizzato per riga FISICA

    const auto t_total0 = std::chrono::steady_clock::now();
    (void)t_total0; // usiamo MPI_Wtime per i timer fini, steady_clock non serve qui

    const double t_start = MPI_Wtime();

    #pragma omp parallel default(none) \
        shared(L, n, seed, plan, rank, chunk_size, norm_chunk_size, final_vector_out, \
               timers, result, row_shift, shift_rows, x_full, x_next, \
               y_phys_local, y_phys_full)
    {
        #pragma omp single
        {
            // ---- Fase 0: init di x (identica su ogni rank, deterministica) ----
            // Ogni rank genera l'INTERO x con lo stesso seed: e' ridondante
            // (O(n) per rank) ma evita di dover distribuire x a inizio run.
            // Costo trascurabile una tantum, per questo va nel timer "init",
            // separato dal breakdown per-iterazione.
            const double t_init0 = MPI_Wtime();

            SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
            for (double& v : x_full) {
                v = rng.next_unit();
            }

            // Normalizzazione iniziale: qui x_full e' gia' COMPLETO e identico
            // su ogni rank, quindi -- a differenza del pattern per-iterazione --
            // non basta scalare la propria fetta fisica: bisogna scalare tutto
            // x_full (0..n), altrimenti ogni rank avrebbe un x diverso dopo
            // l'operazione. Il dot locale invece si fa sulla PROPRIA fetta
            // fisica per riusare lo stesso pattern local_dot -> Allreduce.
            const double local_sq0 = local_dot_omp_tasks(x_full, x_full,
                                                          L.row_begin, L.num_rows,
                                                          norm_chunk_size);
            double global_sq0 = 0.0;

            #pragma omp taskwait // local_dot_omp_tasks usa taskgroup, quindi e' gia' sincrono; il taskwait qui e' difensivo
            {
                const double t_red0 = MPI_Wtime();
                MPI_Allreduce(&local_sq0, &global_sq0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                timers.reduction_sec += MPI_Wtime() - t_red0;
            }

            const double inv0 = 1.0 / std::sqrt(global_sq0);
            local_scale_omp_tasks(x_full, 0, n, inv0, norm_chunk_size);

            timers.init_sec = MPI_Wtime() - t_init0;

            // ---- Fase 1: loop principale, NUM_ITERS iterazioni ----
            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                const double t_epoch0 = MPI_Wtime();
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    row_shift = (row_shift + shift_rows) % n;
                }
                timers.epoch_transition_sec += MPI_Wtime() - t_epoch0;

                // --- SpMV locale: nessuna comunicazione, opera solo su L e x_full ---
                const double t_comp0 = MPI_Wtime();
                spmv_local_omp_tasks(L, x_full, y_phys_local, chunk_size);
                #pragma omp taskwait // spmv_local_omp_tasks NON ha un taskgroup interno: serve qui esplicito
                timers.local_compute_sec += MPI_Wtime() - t_comp0;

                // --- Norma locale (sulla propria fetta fisica) + riduzione globale ---
                const double t_comp1 = MPI_Wtime();
                const double local_sq = local_dot_omp_tasks(y_phys_local, y_phys_local,
                                                             0, L.num_rows, norm_chunk_size);
                timers.local_compute_sec += MPI_Wtime() - t_comp1;

                double global_sq = 0.0;
                const double t_red1 = MPI_Wtime();
                MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                timers.reduction_sec += MPI_Wtime() - t_red1;

                // --- Scala SOLO la fetta locale (decisione #1: massima localita') ---
                const double t_comp2 = MPI_Wtime();
                const double inv = 1.0 / std::sqrt(global_sq);
                local_scale_omp_tasks(y_phys_local, 0, L.num_rows, inv, norm_chunk_size);
                timers.local_compute_sec += MPI_Wtime() - t_comp2;

                // --- Allgatherv: ricompone il vettore fisico completo su ogni rank ---
                const double t_comm0 = MPI_Wtime();
                gather_full_vector(y_phys_local, y_phys_full, plan);
                timers.communication_sec += MPI_Wtime() - t_comm0;

                // --- Permutazione: riordino locale fisico -> logico, zero rete ---
                const double t_comp3 = MPI_Wtime();
                apply_shift_permutation_omp_tasks(y_phys_full, x_next, row_shift, n, norm_chunk_size);
                timers.local_compute_sec += MPI_Wtime() - t_comp3;

                x_full.swap(x_next); // O(1), scambio di buffer
            }

            // ---- Fase 2: passo finale per il valore Rayleigh-like ----
            // Stessa sequenza SpMV + Allgatherv + permutazione, MA senza
            // normalizzare (coerente col sequenziale: l'ultimo SpMV serve
            // solo per il dot finale, non per continuare l'iterazione).
            const double t_final0 = MPI_Wtime();
            spmv_local_omp_tasks(L, x_full, y_phys_local, chunk_size);
            #pragma omp taskwait
            timers.local_compute_sec += MPI_Wtime() - t_final0;

            const double t_comm_final = MPI_Wtime();
            gather_full_vector(y_phys_local, y_phys_full, plan);
            timers.communication_sec += MPI_Wtime() - t_comm_final;

            const double t_perm_final = MPI_Wtime();
            apply_shift_permutation_omp_tasks(y_phys_full, x_next, row_shift, n, norm_chunk_size);
            timers.local_compute_sec += MPI_Wtime() - t_perm_final;
            // x_next ora contiene "y" in ordine logico; x_full e' ancora "x"

            // ---- Fase 3: rayleigh + checksum SOLO su rank 0 (decisione #3) ----
            // x_full e x_next sono gia' repliche complete e identiche su ogni
            // rank grazie all'Allgatherv, quindi calcolarli su rank 0 e basta
            // e' corretto e non richiede nessuna comunicazione aggiuntiva.
            if (rank == 0) {
                double rayleigh = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    rayleigh += x_full[i] * x_next[i];
                }
                result.rayleigh = rayleigh;
                result.checksum = checksum_vector(x_full);
                result.final_row_shift = row_shift;

                if (final_vector_out != nullptr) {
                    *final_vector_out = x_full;
                }
            }
        } // fine single (barrier implicita)
    } // fine parallel

    timers.total_sec = MPI_Wtime() - t_start;

    return MpiIterativeResult{result, timers};
}


// ==========================================
// 7. MAIN
// ==========================================

// Riduce un campo di ExecutionTimers su tutti i rank, calcolando sia il
// massimo (il rank piu' lento domina il tempo percepito) sia la media
// (utile per capire quanto e' distribuito lo sbilanciamento). Entrambi i
// valori finiscono nel report per l'analisi "interazione MPI/OpenMP" e
// "bottleneck di scalabilita'" richiesta dalla consegna.
static void reduce_and_print_timer(const char* label, double local_value,
                                   int num_ranks) {
    double max_value = 0.0;
    double sum_value = 0.0;

    MPI_Reduce(&local_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_value, &sum_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        const double avg_value = sum_value / static_cast<double>(num_ranks);
        std::cout << label
                  << " avg=" << avg_value
                  << " max=" << max_value
                  << " imbalance=" << (max_value - avg_value) << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::uint64_t n64 = 0;
    std::uint64_t nz  = 0;
    std::uint64_t seed = 111;
    std::uint64_t chunk_size = 1000;
    std::uint64_t norm_chunk_arg = 0;
    std::string mode;
    std::string dump_vector_path;

    const bool args_ok = read_arg_u64(argc, argv, "-n", n64) &&
                         read_arg_u64(argc, argv, "-nz", nz) &&
                         read_arg_str(argc, argv, "-m", mode);

    if (!args_ok) {
        if (rank == 0) {
            std::cerr << "Uso: " << argv[0]
                      << " -n N -nz K -m regular|irregular [-c CHUNK] [-nc NORM_CHUNK] [-s SEED] [--dump-vector FILE]\n";
        }
        MPI_Finalize();
        return 1;
    }

    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_u64(argc, argv, "-c", chunk_size);
    (void)read_arg_u64(argc, argv, "-nc", norm_chunk_arg);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);

    // Chunk automatico per le operazioni vettoriali locali: dimensionato
    // sui thread OpenMP di QUESTO rank (non sul totale globale di thread
    // nel job, che non avrebbe senso per un chunk locale).
    const int omp_threads = omp_get_max_threads();
    const std::size_t norm_chunk_size = (norm_chunk_arg == 0)
        ? std::max<std::size_t>(1, (n + static_cast<std::size_t>(omp_threads) - 1) / static_cast<std::size_t>(omp_threads))
        : static_cast<std::size_t>(norm_chunk_arg);

    if (rank == 0) {
        std::cout << "SPARSE_ITERATION_MPI_OMP\n";
        std::cout << "MPI ranks: " << num_ranks << " | OMP threads/rank: " << omp_threads << "\n";
        std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";
    }

    try {
        std::vector<BlockRange> all_blocks;
        double generation_sec = 0.0;
        double distribution_sec = 0.0;

        const LocalMatrix L = setup_and_distribute(n, nz, seed, mode, rank, num_ranks,
                                                    all_blocks, generation_sec, distribution_sec);

        const AllgatherPlan plan = build_allgather_plan(all_blocks);

        if (rank == 0) {
            std::cout << "distribution_time_sec=" << distribution_sec << "\n\n";
        }

        std::vector<double> final_vector;
        std::vector<double>* final_vector_out =
            (rank == 0 && !dump_vector_path.empty()) ? &final_vector : nullptr;

        const MpiIterativeResult mpi_result =
            iterative_spmv_evolving_mpi_omp(L, n, seed, plan, rank,
                                            static_cast<std::size_t>(chunk_size),
                                            norm_chunk_size, final_vector_out);

        // Breakdown per-rank ridotto su rank 0 (avg + max, vedi commento sopra)
        reduce_and_print_timer("init_sec",             mpi_result.timers.init_sec,             num_ranks);
        reduce_and_print_timer("local_compute_sec",    mpi_result.timers.local_compute_sec,    num_ranks);
        reduce_and_print_timer("reduction_sec",        mpi_result.timers.reduction_sec,        num_ranks);
        reduce_and_print_timer("communication_sec",    mpi_result.timers.communication_sec,    num_ranks);
        reduce_and_print_timer("epoch_transition_sec", mpi_result.timers.epoch_transition_sec, num_ranks);
        reduce_and_print_timer("total_sec",             mpi_result.timers.total_sec,             num_ranks);

        if (rank == 0) {
            std::cout << std::setprecision(15);
            std::cout << "rayleigh=" << mpi_result.result.rayleigh << "\n";
            std::cout << "checksum=0x" << std::hex << mpi_result.result.checksum << std::dec << "\n";

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Time (sec) = " << mpi_result.timers.total_sec << "\n";

            if (!dump_vector_path.empty()) {
                dump_vector(dump_vector_path, final_vector);
                std::cout << "vector_dump=" << dump_vector_path << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}