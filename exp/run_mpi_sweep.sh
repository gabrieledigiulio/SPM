#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - MPI + OpenMP Hybrid Sweep (8 Nodi)
# ==============================================================================

# ==========================================
# 1. Configurazione Parametri Benchmark
# ==========================================
MPI_OMP_BIN="../mpi_omp_tasks_SpMV" 
SRUN_TIME="00:10:00"

# Matrice Grande per test prestazionali
N=1000000
NZ=25000000
MODE="irregular"
SEED=111
BLOCK_SIZE=1024
REPEATS=3

# Hardware (Fissato per la configurazione massima)
N_NODES=8
CORES_PER_NODE=16

# Ambiente OpenMP
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Output
OUTPUT_DIR="results"
OUTPUT_FILE="${OUTPUT_DIR}/mpi_sweep_results.csv"
mkdir -p "$OUTPUT_DIR"

echo "Nodes,Total_Ranks,Threads_Per_Rank,Block_Size,Total_Time_Med,Comp_Time_Med,Comm_Time_Med,Red_Time_Med,Epoch_Time_Med,Scatt_Time_Med" > "$OUTPUT_FILE"

# ==========================================
# 2. Funzioni di Utilità
# ==========================================
extract_comp_time() { grep -oP '(?:Computation time|Time) \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comm_time() { grep -oP 'Communication time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_red_time()  { grep -oP 'Reduction time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_epoch_time(){ grep -oP 'Epoch transition \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_scatt_time(){ grep -oP 'Scatter time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_time()      { grep -oP 'Time \(sec\) = \K[0-9.]+' | head -1 || true; }

calculate_median() {
    printf "%s\n" "$@" | awk '
    { a[i++] = $1 }
    END {
        n = asort(a);
        if (n % 2 == 1) { print a[(n+1)/2] }
        else { print (a[n/2] + a[(n/2)+1]) / 2.0 }
    }'
}

# ==========================================
# 3. Esecuzione dello Sweep
# ==========================================
echo "=========================================================="
echo " INIZIO HYBRID SWEEP MPI+OpenMP"
echo " Matrice: $N x $NZ | Modalità: $MODE"
echo " Nodi Fisici: $N_NODES | Core per Nodo: $CORES_PER_NODE"
echo "=========================================================="

# Testiamo tutte le combinazioni possibili di processi e thread
for THREADS_PER_RANK in 1 2 4 8 16 32; do
    
    if [ "$THREADS_PER_RANK" -eq 32 ]; then
        RANKS_PER_NODE=1
        EXTRA_SRUN_ARGS="--oversubscribe"
    else
        RANKS_PER_NODE=$(( CORES_PER_NODE / THREADS_PER_RANK ))
        EXTRA_SRUN_ARGS=""
    fi
    
    TOTAL_RANKS=$(( RANKS_PER_NODE * N_NODES ))
    
    echo ">> Test: Ranks=${TOTAL_RANKS} ( ${RANKS_PER_NODE} per nodo ) | Threads/Rank=${THREADS_PER_RANK}"
    
    tot_times=()
    comp_times=()
    comm_times=()
    red_times=()
    epoch_times=()
    scatt_times=()

    for r in $(seq 1 "$REPEATS"); do
        echo "   - Ripetizione $r/$REPEATS..."
        
        OUTPUT_MPI=$(OMP_NUM_THREADS="$THREADS_PER_RANK" \
            srun --time="$SRUN_TIME" --mpi=pmix \
                 -N "$N_NODES" -n "$TOTAL_RANKS" --cpus-per-task="$THREADS_PER_RANK" $EXTRA_SRUN_ARGS \
                 "$MPI_OMP_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS_PER_RANK" -c "$BLOCK_SIZE" -nc "$BLOCK_SIZE")

        tot_times+=($(echo "$OUTPUT_MPI" | extract_time))
        comp_times+=($(echo "$OUTPUT_MPI" | extract_comp_time))
        comm_times+=($(echo "$OUTPUT_MPI" | extract_comm_time))
        red_times+=($(echo "$OUTPUT_MPI" | extract_red_time))
        epoch_times+=($(echo "$OUTPUT_MPI" | extract_epoch_time))
        scatt_times+=($(echo "$OUTPUT_MPI" | extract_scatt_time))
    done

    med_tot=$(calculate_median "${tot_times[@]}")
    med_comp=$(calculate_median "${comp_times[@]}")
    med_comm=$(calculate_median "${comm_times[@]}")
    med_red=$(calculate_median "${red_times[@]}")
    med_epoch=$(calculate_median "${epoch_times[@]}")
    med_scatt=$(calculate_median "${scatt_times[@]}")

    echo "   -> Mediana Tempo Totale: ${med_tot} s"
    echo "----------------------------------------------------------"

    echo "$N_NODES,$TOTAL_RANKS,$THREADS_PER_RANK,$BLOCK_SIZE,$med_tot,$med_comp,$med_comm,$med_red,$med_epoch,$med_scatt" >> "$OUTPUT_FILE"
    
done

echo "Sweep Completato! Risultati salvati in: $OUTPUT_FILE"