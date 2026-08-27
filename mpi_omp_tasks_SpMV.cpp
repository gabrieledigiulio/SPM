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
// Threading model: MPI is initialized with MPI_THREAD_FUNNELED. ALL MPI
// calls (Allreduce, Allgatherv, Bcast, ...) happen from the single OpenMP
// thread that executes the "#pragma omp single" block -- never from inside
// a task. This is required for correctness when mixing MPI with OpenMP
// tasks, and is verified explicitly at startup (see main()).
//
// Istruzioni di compilazione:
//   mpic++ -O3 -std=c++20 -fopenmp -I . -Wall mpi_omp_SpMV.cpp -o mpi_omp_SpMV
//
// Istruzioni di esecuzione:
//   -n   Dimensione della matrice, NxN
//   -nz  Numero totale di elementi non nulli
//   -m   Modalità della matrice: regular o irregular
//   -t   Numero di thread OpenMP per processo (default: valore di OMP_NUM_THREADS)
//   -c   Dimensione del chunk per SpMV locale (default: 1000)
//   -nc  Dimensione del chunk per operazioni vettoriali locali (0 = automatico)
//   -s   Seed opzionale (default: 111)
//   --dump-vector FILE
//        File di output opzionale per il vettore normalizzato finale
//
// Esempio (4 nodi, 4 processi, 4 thread ciascuno):
//   mpirun -np 4 -x OMP_NUM_THREADS=4 ./mpi_omp_SpMV -n 500000 -nz 20000000 -m irregular
// ==============================================================================

#include "matrix_generation.hpp"
#include "utils.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
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
// la generazione non e' parte del "computation time"). Usiamo MPI_Wtime
// per coerenza con tutti gli altri timer del file (MPI_Init e' gia'
// avvenuto quando questa funzione viene chiamata).
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

        const double tg0 = MPI_Wtime();
        try {
            G = generate_matrix(n, nz, seed, mode);
        } catch (const std::exception& e) {
            ok = 0;
            error_msg = e.what();
        }
        generation_sec = MPI_Wtime() - tg0;

        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) {
            throw std::runtime_error("generazione matrice fallita: " + error_msg);
        }

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n";

        all_blocks = partition_rows_by_nnz(G.A, static_cast<std::size_t>(num_ranks));

        // Bcast: tutti i rank devono conoscere il partizionamento completo
        // per calcolare autonomamente i counts/displs fissi dell'Allgatherv
        // usato nel ciclo iterativo.
        MPI_Bcast(all_blocks.data(), num_ranks * 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        const double td0 = MPI_Wtime();

        LocalMatrix local;
        for (int r = 0; r < num_ranks; ++r) {
            LocalMatrix piece = extract_local_matrix(G.A, all_blocks[r]);
            if (r == 0) {
                local = std::move(piece);
            } else {
                send_local_matrix(piece, r);
            }
        }

        distribution_sec = MPI_Wtime() - td0;

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

// Breakdown dei tempi richiesto dal report. Tutti i campi hanno un
// inizializzatore di default (buona pratica difensiva: qualsiasi lettura
// accidentale prima della prima misura restituisce 0.0, non un valore
// indeterminato). Accumulati (+=) dal thread "single": nessuna
// sincronizzazione necessaria, un solo thread li tocca.


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

struct MpiIterativeResult {
    IterativeResult result;   // rayleigh/checksum/final_row_shift -- validi SOLO su rank 0
    ExecutionTimers timers;   // validi su OGNI rank (necessari per l'analisi di imbalance)
};

// L               : sottomatrice fisica locale di questo rank (mai modificata)
// n                : dimensione globale del problema
// plan             : counts/displs fissi per l'Allgatherv, calcolati una volta in main()
// chunk_size       : granularita' dei task SpMV (righe fisiche locali)
// norm_chunk_size  : granularita' dei task per dot/scale/permutazione
// final_vector_out : se non nullptr, riceve x finale COMPLETO -- popolato solo su rank 0
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
    std::vector<double> x_full(n);                // vettore logico completo, replicato su ogni rank
    std::vector<double> x_next(n);                // buffer di destinazione della permutazione
    std::vector<double> y_phys(L.num_rows);        // output SpMV locale, indicizzato per riga FISICA
    std::vector<double> y_phys_full(n);            // dopo l'Allgatherv, indicizzato per riga FISICA

    const double t_start = MPI_Wtime();

    // Nota sul data-sharing: la regione "parallel" qui sotto usa
    // default(shared) invece di default(none). Motivo: le chiamate MPI
    // (MPI_Allreduce, MPI_Allgatherv) referenziano macro come MPI_COMM_WORLD
    // / MPI_DOUBLE / MPI_SUM, che nelle implementazioni MPI (es. Open MPI)
    // espandono a simboli globali interni non standard (es.
    // "ompi_mpi_comm_world"). Elencarli esplicitamente in una clausola
    // shared() legherebbe il codice a un'implementazione MPI specifica,
    // rompendo la portabilita' (es. su MPICH non compilerebbe piu'). Dato
    // che questo livello esterno ha un solo punto di reale esecuzione (il
    // thread "single"), default(shared) qui e' innocuo: il rigore di
    // default(none) resta dove conta davvero, cioe' su ogni "#pragma omp
    // task" qui sotto, dove piu' thread eseguono realmente in parallelo.
    #pragma omp parallel
    {
        #pragma omp single
        {
            // ---- Fase 0: inizializzazione di x (identica al sequenziale) ----
            const double t_init0 = MPI_Wtime();

            // Il flusso RNG deve restare rigorosamente sequenziale (single
            // thread) per generare la stessa sequenza del riferimento.
            SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
            for (double& v : x_full) {
                v = rng.next_unit();
            }

            // Normalizzazione iniziale: x_full e' GIA' identico e completo
            // su ogni rank (stesso seed, stesso flusso RNG deterministico),
            // quindi la riduzione e' puramente locale -- zero comunicazione,
            // a differenza del pattern per-iterazione dove i dati sono
            // fisicamente distribuiti e serve un vero MPI_Allreduce.
            {
                const double sumsq = local_dot_omp_tasks(x_full, x_full, 0, n, norm_chunk_size);
                const double inv = 1.0 / std::sqrt(sumsq);
                local_scale_omp_tasks(x_full, 0, n, inv, norm_chunk_size);
            }

            timers.init_sec = MPI_Wtime() - t_init0;

            // ---- Fase 1: loop principale, NUM_ITERS iterazioni ----
            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    const double te0 = MPI_Wtime();
                    row_shift = (row_shift + shift_rows) % n;
                    timers.epoch_transition_sec += MPI_Wtime() - te0;
                }

                // (a) SpMV locale: produce y_phys, indicizzato per riga fisica.
                const double tc0 = MPI_Wtime();
                spmv_local_omp_tasks(L, x_full, y_phys, chunk_size);
                #pragma omp taskwait
                const double spmv_elapsed = MPI_Wtime() - tc0;
                timers.computation_sec += spmv_elapsed;
                timers.spmv_sec += spmv_elapsed;

                // (b) Somma dei quadrati locale (sulla propria fetta fisica).
                const double tc1 = MPI_Wtime();
                const double sumsq_local = local_dot_omp_tasks(y_phys, y_phys, 0, L.num_rows, norm_chunk_size);
                const double dot_elapsed = MPI_Wtime() - tc1;
                timers.computation_sec += dot_elapsed;
                timers.vector_ops_sec += dot_elapsed;

                // (c) Allreduce globale: comunicazione minuscola, un solo double.
                double sumsq_global = 0.0;
                const double tr0 = MPI_Wtime();
                MPI_Allreduce(&sumsq_local, &sumsq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                timers.reduction_sec += MPI_Wtime() - tr0;

                const double inv = 1.0 / std::sqrt(sumsq_global);

                // (d) Normalizza SOLO la fetta locale (piu' efficiente che
                // farlo dopo il gather, che sarebbe O(n) ridondante ovunque).
                const double tc2 = MPI_Wtime();
                local_scale_omp_tasks(y_phys, 0, L.num_rows, inv, norm_chunk_size);
                const double scale_elapsed = MPI_Wtime() - tc2;
                timers.computation_sec += scale_elapsed;
                timers.vector_ops_sec += scale_elapsed;

                // (e) Allgatherv: assembla il vettore fisico completo.
                const double tcm0 = MPI_Wtime();
                gather_full_vector(y_phys, y_phys_full, plan);
                timers.communication_sec += MPI_Wtime() - tcm0;

                // (f) Permutazione locale: applica row_shift, produce x_next.
                const double tc3 = MPI_Wtime();
                apply_shift_permutation_omp_tasks(y_phys_full, x_next, row_shift, n, chunk_size);
                x_full.swap(x_next);
                const double scatter_elapsed = MPI_Wtime() - tc3;
                timers.computation_sec += scatter_elapsed;
                timers.scatter_sec += scatter_elapsed;
            }

            // ---- Fase 2: passo finale per il valore Rayleigh-like ----
            // Stessa sequenza SpMV + Allgatherv, MA senza normalizzare
            // (coerente col sequenziale: l'ultimo SpMV serve solo per il
            // dot finale, non per continuare l'iterazione).
            //
            // spmv_local e gather_full_vector sono chiamate collettive/di
            // gruppo: DEVONO essere eseguite da ogni rank, anche se solo il
            // rank 0 user\a poi il risultato (altrimenti l'Allgatherv si
            // blocca in attesa di rank che non arrivano mai alla chiamata).
            const double tf0 = MPI_Wtime();
            spmv_local_omp_tasks(L, x_full, y_phys, chunk_size);
            #pragma omp taskwait
            const double final_spmv_elapsed = MPI_Wtime() - tf0;
            timers.computation_sec += final_spmv_elapsed;
            timers.spmv_sec += final_spmv_elapsed;

            const double tfc0 = MPI_Wtime();
            gather_full_vector(y_phys, y_phys_full, plan);
            timers.communication_sec += MPI_Wtime() - tfc0;

            // La permutazione finale e il calcolo di rayleigh/checksum sono
            // invece puro lavoro LOCALE (non collettivo): possiamo
            // limitarli al solo rank 0, che e' l'unico a doverne fare uso
            // (stampa risultati, eventuale dump del vettore).
            if (rank == 0) {
                const double tp0 = MPI_Wtime();
                apply_shift_permutation_omp_tasks(y_phys_full, x_next, row_shift, n, chunk_size);
                const double final_scatter_elapsed = MPI_Wtime() - tp0;
                timers.computation_sec += final_scatter_elapsed;
                timers.scatter_sec += final_scatter_elapsed;
                // x_next ora contiene "y" (logico) = A_shifted * x_full;
                // x_full resta il vettore finale da riportare (NON swap qui).

                const double tdot0 = MPI_Wtime();
                result.rayleigh = local_dot_omp_tasks(x_full, x_next, 0, n, norm_chunk_size);
                const double final_dot_elapsed = MPI_Wtime() - tdot0;
                timers.computation_sec += final_dot_elapsed;
                timers.vector_ops_sec += final_dot_elapsed;
                const double tchk0 = MPI_Wtime();
                result.checksum = checksum_vector(x_full);
                const double chk_elapsed = MPI_Wtime() - tchk0;
                timers.computation_sec += chk_elapsed;
                timers.vector_ops_sec += chk_elapsed;
                result.final_row_shift = row_shift;

                if (final_vector_out != nullptr) {
                    *final_vector_out = x_full;
                }
            }

            timers.total_sec = MPI_Wtime() - t_start;
        } // fine single (barrier implicita in uscita)
    } // fine parallel

    return MpiIterativeResult{result, timers};
}

// ==========================================
// 7. MAIN
// ==========================================

// Riduce un campo di ExecutionTimers su tutti i rank, calcolando sia il
// massimo (il rank piu' lento domina il tempo percepito) sia la media
// (utile per capire quanto e' distribuito lo sbilanciamento). Entrambi i
// valori finiscono nel report per l'analisi "interazione MPI/OpenMP" e
// "bottleneck di scalabilita'" richiesta dalla consegna. Ritorna il
// massimo, cosi' il chiamante puo' riusarlo (es. per il "Time (sec) ="
// finale, nello stesso formato delle altre implementazioni del progetto).
static double reduce_and_print_timer(const char* label, double local_value,
                                     int num_ranks, int rank) {
    double max_value = 0.0;
    double sum_value = 0.0;

    MPI_Reduce(&local_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_value, &sum_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double avg_value = sum_value / static_cast<double>(num_ranks);
        std::cout << label << " = " << max_value << "\n";
        std::cout << "  " << label
                  << " avg=" << avg_value
                  << " max=" << max_value
                  << " imbalance=" << (max_value - avg_value) << "\n";
    }

    return max_value;
}

int main(int argc, char** argv) {
    // MPI_THREAD_FUNNELED: richiesto esplicitamente perche' mescoliamo
    // MPI e task OpenMP. Tutte le chiamate MPI avvengono dal thread
    // "single" (mai da dentro un task), quindi FUNNELED e' sufficiente
    // (non serve il piu' costoso MPI_THREAD_MULTIPLE). Verifichiamo che
    // l'implementazione lo conceda davvero, invece di assumerlo.
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "Error: the MPI implementation does not support MPI_THREAD_FUNNELED "
                      << "(required to safely mix MPI and OpenMP tasks)\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t threads_arg, chunk_size, norm_chunk_arg;
    std::string mode;
    std::string dump_vector_path;

    // argv e' identico su ogni rank (mpirun lo replica), quindi ogni
    // processo puo' fare il parsing in modo indipendente, senza broadcast.
    // Tutti i parametri tranne il seed sono nella condizione obbligatoria
    const bool args_ok = read_arg_u64(argc, argv, "-n", n64) &&
                         read_arg_u64(argc, argv, "-nz", nz) &&
                         read_arg_str(argc, argv, "-m", mode) &&
                         read_arg_u64(argc, argv, "-t", threads_arg) &&
                         read_arg_u64(argc, argv, "-c", chunk_size) &&
                         read_arg_u64(argc, argv, "-nc", norm_chunk_arg);

    if (!args_ok) {
        if (rank == 0) {
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Lettura opzionale del seed e del dump
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);

    if (threads_arg > 0) {
        omp_set_num_threads(static_cast<int>(threads_arg));
    }
    const int omp_threads = omp_get_max_threads();

    const std::size_t norm_chunk_size = (norm_chunk_arg == 0)
        ? std::max<std::size_t>(1, (n + static_cast<std::size_t>(omp_threads) - 1) / static_cast<std::size_t>(omp_threads))
        : static_cast<std::size_t>(norm_chunk_arg);

    if (rank == 0) {
        std::cout << "SPARSE_ITERATION_MPI_OMP\n";
        std::cout << "MPI Ranks: " << num_ranks << " | OpenMP Threads/rank: " << omp_threads << "\n";
        std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";
    }

    try {
        std::vector<BlockRange> all_blocks;
        double generation_sec = 0.0;
        double distribution_sec = 0.0;

        const LocalMatrix L = setup_and_distribute(n, nz, seed, mode, rank, num_ranks,
                                                    all_blocks, generation_sec, distribution_sec);

        if (rank == 0) {
            std::cout << "distribution_time_sec=" << distribution_sec << "\n\n";
        }

        const AllgatherPlan plan = build_allgather_plan(all_blocks);

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out =
            (rank == 0 && !dump_vector_path.empty()) ? &final_vector : nullptr;

        // Allinea tutti i rank prima di avviare il timer totale: senza
        // questa barriera, rank che finiscono il setup in tempi diversi
        // (es. rank 0 che ha appena finito di spedire i dati agli altri)
        // partirebbero con orologi sfalsati, distorcendo l'attribuzione
        // del tempo nella primissima iterazione (tra "reduction"/"comm"
        // di attesa e "local_computation" vero).
        MPI_Barrier(MPI_COMM_WORLD);

        const MpiIterativeResult mpi_result =
            iterative_spmv_evolving_mpi_omp(L, n, seed, plan, rank,
                                            static_cast<std::size_t>(chunk_size),
                                            norm_chunk_size, final_vector_out);

        if (rank == 0) {
            std::cout << "Time breakdown (seconds):\n";
        }
        reduce_and_print_timer("Init time (sec)", mpi_result.timers.init_sec, num_ranks, rank);
        reduce_and_print_timer("Computation time (sec)", mpi_result.timers.computation_sec, num_ranks, rank);
        reduce_and_print_timer("SpMV time (sec)", mpi_result.timers.spmv_sec, num_ranks, rank);
        reduce_and_print_timer("Vector ops time (sec)", mpi_result.timers.vector_ops_sec, num_ranks, rank);
        reduce_and_print_timer("Scatter time (sec)", mpi_result.timers.scatter_sec, num_ranks, rank);
        reduce_and_print_timer("Reduction time (sec)", mpi_result.timers.reduction_sec, num_ranks, rank);
        reduce_and_print_timer("Communication time (sec)", mpi_result.timers.communication_sec, num_ranks, rank);
        reduce_and_print_timer("Epoch transition (sec)", mpi_result.timers.epoch_transition_sec, num_ranks, rank);
        const double total_sec_max =
            reduce_and_print_timer("Total time (sec)", mpi_result.timers.total_sec, num_ranks, rank);

        if (rank == 0) {
            std::cout << std::setprecision(15);
            std::cout << "rayleigh=" << mpi_result.result.rayleigh << "\n";
            std::cout << "checksum=0x" << std::hex << mpi_result.result.checksum << std::dec << "\n";

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Time (sec) = " << total_sec_max << "\n";

            if (!dump_vector_path.empty()) {
                dump_vector(dump_vector_path, final_vector);
                std::cout << "vector_dump=" << dump_vector_path << "\n";
            }
        }
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // evita che gli altri rank restino bloccati
        return 1;
    }

    MPI_Finalize();
    return 0;
}