#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Benchmark Granularità con Mediana e Ripetizioni
# ==============================================================================

# Parametri del problema "Largo" del report
N=1000000
NZ=25000000
MODE="irregular"
SEED=111

# Configurazione delle risorse e delle ripetizioni
THREADS=16
MPI_NODES=8
MPI_RANKS=8
RANKS_PER_NODE=$(( MPI_RANKS / MPI_NODES ))
MPIRUN_EXTRA_ARGS="--map-by ppr:${RANKS_PER_NODE}:node"

REPEATS=3
BLOCK_SIZES=(256 512 1024 2048 4096 8192 16384)

# Creazione della cartella results
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

CSV_FILE="$RESULTS_DIR/granularity_unified_results.csv"

# Inizializzazione del CSV (con la colonna per la mediana del tempo totale)
echo "Implementation,ChunkSize,Threads,MedianTotalTime,MedianSpMVTime,MedianVectorOpsTime" > "$CSV_FILE"

# Funzioni di estrazione
extract_comp_time()   { grep -oP '(?:Computation time|Time) \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time()   { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_vecops_time() { grep -oP 'Vector ops time \(sec\) = \K[0-9.]+' | head -1 || true; }

# Funzione per calcolare la mediana di un array di numeri float/double usando awk
calculate_median() {
    local arr=("$@")
    # Ordina l'array numericamente e stampa il valore mediano
    printf '%s\n' "${arr[@]}" | sort -n | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) print 0;
            else if (NR % 2 == 1) print a[(NR + 1) / 2];
            else print (a[NR / 2] + a[NR / 2 + 1]) / 2;
        }
    '
}

# ==========================================
# 1. Benchmark C++ Threads (ThreadPool)
# ==========================================
echo "=========================================================="
echo " Avvio Granularità: C++ Threads (Repeats: $REPEATS)"
echo "=========================================================="

for chunk in "${BLOCK_SIZES[@]}"; do
    echo ">> Testando C++ Threads con Chunk = $chunk ..."
    tot_times=()
    spmv_times=()
    vec_times=()

    for r in $(seq 1 "$REPEATS"); do
        out=$(../threadpool_SpMV -n "$N" -nz "$NZ" -m "$MODE" -t "$THREADS" -c "$chunk" -nc "$chunk" -s "$SEED")
        tot_times+=($(echo "$out" | extract_comp_time))
        spmv_times+=($(echo "$out" | extract_spmv_time))
        vec_times+=($(echo "$out" | extract_vecops_time))
    done

    med_tot=$(calculate_median "${tot_times[@]}")
    med_spmv=$(calculate_median "${spmv_times[@]}")
    med_vec=$(calculate_median "${vec_times[@]}")

    echo "CPP_THREADS,$chunk,$THREADS,$med_tot,$med_spmv,$med_vec" >> "$CSV_FILE"
    echo "  -> Mediana Totale: ${med_tot}s (SpMV: ${med_spmv}s, VectorOps: ${med_vec}s)"
done

# ==========================================
# 2. Benchmark OpenMP Tasks
# ==========================================
echo "=========================================================="
echo " Avvio Granularità: OpenMP Tasks (Repeats: $REPEATS)"
echo "=========================================================="

for chunk in "${BLOCK_SIZES[@]}"; do
    echo ">> Testando OpenMP Tasks con Chunk = $chunk ..."
    tot_times=()
    spmv_times=()
    vec_times=()

    for r in $(seq 1 "$REPEATS"); do
        out=$(../omp_tasks_SpMV -n "$N" -nz "$NZ" -m "$MODE" -t "$THREADS" -c "$chunk" -nc "$chunk" -s "$SEED")
        tot_times+=($(echo "$out" | extract_comp_time))
        spmv_times+=($(echo "$out" | extract_spmv_time))
        vec_times+=($(echo "$out" | extract_vecops_time))
    done

    med_tot=$(calculate_median "${tot_times[@]}")
    med_spmv=$(calculate_median "${spmv_times[@]}")
    med_vec=$(calculate_median "${vec_times[@]}")

    echo "OMP_TASKS,$chunk,$THREADS,$med_tot,$med_spmv,$med_vec" >> "$CSV_FILE"
    echo "  -> Mediana Totale: ${med_tot}s (SpMV: ${med_spmv}s, VectorOps: ${med_vec}s)"
done

# ==========================================
# 3. Benchmark MPI + OpenMP
# ==========================================
echo "=========================================================="
echo " Avvio Granularità: MPI + OpenMP (Repeats: $REPEATS)"
echo "=========================================================="

for chunk in "${BLOCK_SIZES[@]}"; do
    echo ">> Testando MPI+OpenMP con Chunk = $chunk ..."
    tot_times=()
    spmv_times=()
    vec_times=()

    for r in $(seq 1 "$REPEATS"); do
        out=$(OMP_NUM_THREADS="$THREADS" \
              mpirun -np "$MPI_RANKS" $MPIRUN_EXTRA_ARGS ../mpi_omp_SpMV \
              -n "$N" -nz "$NZ" -m "$MODE" -c "$chunk" -nc "$chunk" -s "$SEED")
        tot_times+=($(echo "$out" | extract_comp_time))
        spmv_times+=($(echo "$out" | extract_spmv_time))
        vec_times+=($(echo "$out" | extract_vecops_time))
    done

    med_tot=$(calculate_median "${tot_times[@]}")
    med_spmv=$(calculate_median "${spmv_times[@]}")
    med_vec=$(calculate_median "${vec_times[@]}")

    echo "MPI_OMP,$chunk,$THREADS,$med_tot,$med_spmv,$med_vec" >> "$CSV_FILE"
    echo "  -> Mediana Totale: ${med_tot}s (SpMV: ${med_spmv}s, VectorOps: ${med_vec}s)"
done

echo "=========================================================="
echo " Benchmark Granularità completato con successo!"
echo " Risultati salvati in: $CSV_FILE"
echo "=========================================================="