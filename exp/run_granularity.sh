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

# Parametri di esecuzione
THREADS=16
REPEATS=3
BLOCK_SIZES=(256 512 1024 2048 4096 8192 16384)

# Parametri MPI (attivi solo se MODE_TYPE == mpi)
MPI_NODES=8
MPI_RANKS=8
RANKS_PER_NODE=$(( MPI_NODES > 0 ? MPI_RANKS / MPI_NODES : 1 ))
MPIRUN_EXTRA_ARGS="--map-by ppr:${RANKS_PER_NODE}:node"

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# CSV specifico in base alla modalità scelta
CSV_FILE="$RESULTS_DIR/results_granularity_${MODE_TYPE}.csv"

# ==============================================================================
# RECAP CONFIGURAZIONE SCELTA
# ==============================================================================
echo "=========================================================="
echo " SPM PROJECT - CONFIGURAZIONE BENCHMARK GRANULARITÀ"
echo "=========================================================="
echo " Modalità testata      : $MODE_TYPE"
echo " Matrice (N x NZ)      : $N righe, $NZ non-zeri"
echo " Tipo di matrice       : $MODE (Seed: $SEED)"
echo " Thread per processo   : $THREADS"
echo " Ripetizioni (Mediana) : $REPEATS"
echo " Chunk Sizes SpMV (-c) : ${BLOCK_SIZES[*]}"
echo " Chunk Sizes Norm (-nc): ${BLOCK_SIZES[*]}"
if [ "$MODE_TYPE" == "mpi" ]; then
echo " --- Parametri MPI ---"
echo " Nodi MPI              : $MPI_NODES"
echo " Rank MPI totali       : $MPI_RANKS"
echo " Rank per nodo         : $RANKS_PER_NODE"
fi
echo " File di output        : $CSV_FILE"
echo "=========================================================="
echo ""

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
        echo "----------------------------------------------------------"
        echo " Avvio esecuzione: C++ Threads (ThreadPool)"
        echo "----------------------------------------------------------"
        for chunk in "${BLOCK_SIZES[@]}"; do
            echo ">> Testando C++ Threads con Chunk SpMV/Norm = $chunk ..."
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
            echo "  -> Mediana Totale: ${med_tot}s (SpMV: ${med_spmv}s, VectorOps: ${med_vec}s)"
        done
        ;;
    omp)
        echo "----------------------------------------------------------"
        echo " Avvio esecuzione: OpenMP Tasks"
        echo "----------------------------------------------------------"
        for chunk in "${BLOCK_SIZES[@]}"; do
            echo ">> Testando OpenMP Tasks con Chunk SpMV/Norm = $chunk ..."
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
            echo "  -> Mediana Totale: ${med_tot}s (SpMV: ${med_spmv}s, VectorOps: ${med_vec}s)"
        done
        ;;
    mpi)
        echo "----------------------------------------------------------"
        echo " Avvio esecuzione: MPI + OpenMP (Nodi: $MPI_NODES, Ranks: $MPI_RANKS)"
        echo "----------------------------------------------------------"
        for chunk in "${BLOCK_SIZES[@]}"; do
            echo ">> Testando MPI+OpenMP con Chunk SpMV/Norm = $chunk ..."
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
            echo "  -> Mediana Totale: ${med_tot}s (SpMV: ${med_spmv}s, VectorOps: ${med_vec}s)"
        done
        ;;
    *)
        echo "Errore: Opzione non valida '$MODE_TYPE'. Usa: threads, omp, oppure mpi."
        exit 1
        ;;
esac

echo "=========================================================="
echo " Test '$MODE_TYPE' completato con successo!"
echo " Risultati salvati in: $CSV_FILE"
echo "=========================================================="