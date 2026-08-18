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
NZ=250000000
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

# Se il file non esiste, creiamo l'intestazione allargata a tutti i componenti
if [ ! -f "$CSV_FILE" ]; then
    echo "Implementation,ChunkSize,Threads,MedianTotalTime,MedianSpMVTime,MedianVectorOpsTime,MedianEpochTime,MedianInitTime" > "$CSV_FILE"
fi

# Funzioni di estrazione compatibili con macOS (BSD grep/sed) e Linux per ogni timer del codice
extract_comp_time()   { grep -oE 'Time \(sec\) = [0-9.]+' | head -1 | awk '{print $NF}' || true; }
extract_spmv_time()   { grep -oE 'SpMV time \(sec\) = [0-9.]+' | head -1 | awk '{print $NF}' || true; }
extract_vecops_time() { grep -oE 'Vector ops time \(sec\) = [0-9.]+' | head -1 | awk '{print $NF}' || true; }
extract_epoch_time()  { grep -oE 'Epoch transition \(sec\) = [0-9.]+' | head -1 | awk '{print $NF}' || true; }
extract_init_time()   { grep -oE 'Init time \(sec\) = [0-9.]+' | head -1 | awk '{print $NF}' || true; }

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
        IMPL_LABEL="CPP_THREADS"
        BIN_PATH="../threadpool_SpMV"
        ;;
    omp)
        IMPL_LABEL="OMP_TASKS"
        BIN_PATH="../omp_tasks_SpMV"
        ;;
    mpi)
        IMPL_LABEL="MPI_OMP"
        BIN_PATH="../mpi_omp_SpMV"
        ;;
    *)
        echo "Errore: Opzione non valida '$MODE_TYPE'. Usa: threads, omp, oppure mpi."
        exit 1
        ;;
esac

echo "----------------------------------------------------------"
echo " Avvio esecuzione: $IMPL_LABEL"
echo "----------------------------------------------------------"

for chunk in "${BLOCK_SIZES[@]}"; do
    echo ">> Testando con Chunk SpMV/Norm = $chunk ..."
    tot_times=()
    spmv_times=()
    vec_times=()
    epoch_times=()
    init_times=()
    
    for r in $(seq 1 "$REPEATS"); do
        if [ "$MODE_TYPE" == "mpi" ]; then
            out=$(OMP_NUM_THREADS="$THREADS" \
                  mpirun -np "$MPI_RANKS" $MPIRUN_EXTRA_ARGS "$BIN_PATH" \
                  -n "$N" -nz "$NZ" -m "$MODE" -c "$chunk" -nc "$chunk" -s "$SEED")
        else
            out=$("$BIN_PATH" -n "$N" -nz "$NZ" -m "$MODE" -t "$THREADS" -c "$chunk" -nc "$chunk" -s "$SEED")
        fi
        
        t_tot=$(echo "$out" | extract_comp_time)
        t_spmv=$(echo "$out" | extract_spmv_time)
        t_vec=$(echo "$out" | extract_vecops_time)
        t_epoch=$(echo "$out" | extract_epoch_time)
        t_init=$(echo "$out" | extract_init_time)
        
        [[ -n "$t_tot" ]] && tot_times+=("$t_tot")
        [[ -n "$t_spmv" ]] && spmv_times+=("$t_spmv")
        [[ -n "$t_vec" ]] && vec_times+=("$t_vec")
        [[ -n "$t_epoch" ]] && epoch_times+=("$t_epoch")
        [[ -n "$t_init" ]] && init_times+=("$t_init")
    done

    if [ ${#tot_times[@]} -gt 0 ]; then
        med_tot=$(calculate_median "${tot_times[@]}")
        med_spmv=$(calculate_median "${spmv_times[@]}")
        med_vec=$(calculate_median "${vec_times[@]}")
        med_epoch=$(calculate_median "${epoch_times[@]}")
        med_init=$(calculate_median "${init_times[@]}")
    else
        med_tot=0; med_spmv=0; med_vec=0; med_epoch=0; med_init=0
    fi

    echo "$IMPL_LABEL,$chunk,$THREADS,$med_tot,$med_spmv,$med_vec,$med_epoch,$med_init" >> "$CSV_FILE"
    echo "  -> Totale: ${med_tot}s [ SpMV: ${med_spmv}s | VectorOps: ${med_vec}s | Epoch: ${med_epoch}s | Init: ${med_init}s ]"
done

echo "=========================================================="
echo " Test '$MODE_TYPE' completato con successo!"
echo " Risultati salvati in: $CSV_FILE"
echo "=========================================================="