#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM Project - Benchmark Granularità Modulare (threads, omp, mpi)
# ==============================================================================

MODE_TYPE="${1:-}"

if [[ -z "$MODE_TYPE" ]]; then
    echo "Errore: Devi specificare una modalità!"
    echo "Uso: ./run_granularity.sh [threads|omp|mpi]"
    exit 1
fi

# Parametri del problema
N=1000000
NZ=25000000
MODE="irregular"
SEED=111

THREADS=16
REPEATS=3
BLOCK_SIZES=(256 512 1024 2048 4096 8192 16384)

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
CSV_FILE="$RESULTS_DIR/granularity_unified_results.csv"

# Se il file non esiste, creiamo l'intestazione
if [ ! -f "$CSV_FILE" ]; then
    echo "Implementation,ChunkSize,Threads,MedianTotalTime,MedianSpMVTime,MedianVectorOpsTime" > "$CSV_FILE"
fi

extract_comp_time()   { grep -oP '(?:Computation time|Time) \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time()   { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_vecops_time() { grep -oP 'Vector ops time \(sec\) = \K[0-9.]+' | head -1 || true; }

calculate_median() {
    local arr=("$@")
    printf '%s\n' "${arr[@]}" | sort -n | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) print 0;
            else if (NR % 2 == 1) print a[(NR + 1) / 2];
            else print (a[NR / 2] + a[NR / 2 + 1]) / 2;
        }
    '
}

case "$MODE_TYPE" in
    threads)
        echo "=========================================================="
        echo " Avvio Granularità: C++ Threads (Repeats: $REPEATS)"
        echo "=========================================================="
        for chunk in "${BLOCK_SIZES[@]}"; do
            echo ">> Testando C++ Threads con Chunk = $chunk ..."
            tot_times=(); spmv_times=(); vec_times=()
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
            echo "  -> Mediana Totale: ${med_tot}s"
        done
        ;;
    omp)
        echo "=========================================================="
        echo " Avvio Granularità: OpenMP Tasks (Repeats: $REPEATS)"
        echo "=========================================================="
        for chunk in "${BLOCK_SIZES[@]}"; do
            echo ">> Testando OpenMP Tasks con Chunk = $chunk ..."
            tot_times=(); spmv_times=(); vec_times=()
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
            echo "  -> Mediana Totale: ${med_tot}s"
        done
        ;;
    mpi)
        MPI_NODES=8
        MPI_RANKS=8
        RANKS_PER_NODE=$(( MPI_RANKS / MPI_NODES ))
        MPIRUN_EXTRA_ARGS="--map-by ppr:${RANKS_PER_NODE}:node"

        echo "=========================================================="
        echo " Avvio Granularità: MPI + OpenMP (Nodes: $MPI_NODES, Repeats: $REPEATS)"
        echo "=========================================================="
        for chunk in "${BLOCK_SIZES[@]}"; do
            echo ">> Testando MPI+OpenMP con Chunk = $chunk ..."
            tot_times=(); spmv_times=(); vec_times=()
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
            echo "  -> Mediana Totale: ${med_tot}s"
        done
        ;;
    *)
        echo "Errore: Opzione non valida '$MODE_TYPE'. Usa: threads, omp, oppure mpi."
        exit 1
        ;;
esac

echo "=========================================================="
echo " Test '$MODE_TYPE' completato! Dati salvati in: $CSV_FILE"
echo "=========================================================="