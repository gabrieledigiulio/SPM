#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Weak Scalability Experiment
# ==============================================================================

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
MPI_OMP_BIN="../mpi_omp_tasks_SpMV"

SRUN_TIME="00:15:00"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Problem Parameters (Base per 1 Nodo)
# Se N_BASE=125000 e NZ_BASE=31250000, 
# a 8 Nodi avremo N=1000000 e NZ=250000000 (il nostro caso d'uso principale)
BASE_N=125000
BASE_NZ=31250000
MODE="irregular"
SEED=111

# Topologia Hardware Ottimizzata
THREADS=16
SPMV_CHUNK=1024
NORM_CHUNK=16384

# Parametri dello Sweep di Scalabilità
NODES_LIST=(1 2 4 8)
REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/weak_scaling_results.csv"

# ==========================================
# 2. Extraction & Math Functions
# ==========================================
extract_tot_time()   { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comp_time()  { grep -oP 'Computation time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comm_time()  { grep -oP 'Communication time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_red_time()   { grep -oP 'Reduction time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_scatt_time() { grep -oP 'Scatter time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_epoch_time() { grep -oP 'Epoch transition \(sec\) = \K[0-9.]+' | head -1 || true; }

# Funzione per calcolare la mediana
calculate_median() {
    local vals=()
    for v in "$@"; do
        if [[ -n "$v" && "$v" != "N/A" ]]; then
            vals+=("$v")
        fi
    done
    
    if [ ${#vals[@]} -eq 0 ]; then
        echo "N/A"
        return
    fi

    printf "%s\n" "${vals[@]}" | sort -n | awk '
        { a[i++] = $1 }
        END {
            if (i % 2 == 1) print a[int(i/2)]
            else print (a[i/2-1] + a[i/2]) / 2.0
        }'
}

# ==========================================
# 3. Execution Logic
# ==========================================
echo "=========================================================="
echo " WEAK SCALABILITY EXPERIMENT ($REPEATS REPEATS)"
echo "=========================================================="
echo "  Base Matrix / Node:   $BASE_N x $BASE_NZ"
echo "  Threads per Rank:     $THREADS"
echo "  SpMV Chunk Size:      $SPMV_CHUNK"
echo "  Norm Chunk Size:      $NORM_CHUNK"
echo "  Nodes Configuration:  ${NODES_LIST[*]}"
echo "=========================================================="

echo "Nodes,N_Global,NZ_Global,Total_Time_Med,Comp_Time_Med,Comm_Time_Med,Red_Time_Med,Scatt_Time_Med,Epoch_Time_Med" > "$CSV_OUTPUT"

for nodes in "${NODES_LIST[@]}"; do
    # Calcolo della dimensione del problema per questa iterazione
    MATRIX_N=$((BASE_N * nodes))
    MATRIX_NZ=$((BASE_NZ * nodes))

    echo "=========================================================="
    echo ">> Testing Config: $nodes Nodes | N=$MATRIX_N, NZ=$MATRIX_NZ"
    echo "=========================================================="

    # Inizializzazione array di timing
    tot_times=()
    comp_times=()
    comm_times=()
    red_times=()
    scatt_times=()
    epoch_times=()

    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"
        
        # --- Esecuzione MPI ---
        out_mpi=$(OMP_NUM_THREADS="$THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$nodes" -n "$nodes" \
            --cpus-per-task="$THREADS" "$MPI_OMP_BIN" -n "$MATRIX_N" -nz "$MATRIX_NZ" -m "$MODE" -s "$SEED" \
            -t "$THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        
        tot_times+=($(echo "$out_mpi" | extract_tot_time))
        comp_times+=($(echo "$out_mpi" | extract_comp_time))
        comm_times+=($(echo "$out_mpi" | extract_comm_time))
        red_times+=($(echo "$out_mpi" | extract_red_time))
        scatt_times+=($(echo "$out_mpi" | extract_scatt_time))
        epoch_times+=($(echo "$out_mpi" | extract_epoch_time))
    done

    # --- Calcolo mediane ---
    m_tot=$(calculate_median "${tot_times[@]}")
    m_comp=$(calculate_median "${comp_times[@]}")
    m_comm=$(calculate_median "${comm_times[@]}")
    m_red=$(calculate_median "${red_times[@]}")
    m_scatt=$(calculate_median "${scatt_times[@]}")
    m_epoch=$(calculate_median "${epoch_times[@]}")

    # Salvataggio su CSV
    echo "$nodes,$MATRIX_N,$MATRIX_NZ,$m_tot,$m_comp,$m_comm,$m_red,$m_scatt,$m_epoch" >> "$CSV_OUTPUT"
    echo "  -> Medians: Tot=${m_tot}s, Comp=${m_comp}s, Comm=${m_comm}s"
done

echo "=========================================================="
echo " Weak scalability sweep completed!"
echo " Results successfully saved to: $CSV_OUTPUT"
echo "=========================================================="