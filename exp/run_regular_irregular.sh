#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Regular vs Irregular Pattern Test
# ==============================================================================

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
PTH_BIN="../pthreads_SpMV"
OMP_BIN="../omp_tasks_SpMV" # Cambia in "../omp_SpMV" se il nome è diverso
MPI_BIN="../mpi_omp_tasks_SpMV"

SRUN_TIME="00:15:00"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Problem Parameters
N=1000000
NZ=250000000
SEED=111

# Topologia Hardware Ottimizzata
THREADS=16
MPI_NODES=8

# Chunk Ottimali
SPMV_CHUNK=1024
NORM_CHUNK=16384

REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/regular_vs_irregular.csv"

# ==========================================
# 2. Extraction & Math Functions
# ==========================================
extract_tot_time()  { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comp_time() { grep -oP 'Computation time \(sec\) = \K[0-9.]+' | head -1 || true; }

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
echo " REGULAR vs IRREGULAR PATTERN EXPERIMENT"
echo "=========================================================="
echo "  Matrix:         $N x $NZ"
echo "  Threads:        $THREADS per node"
echo "  MPI Nodes:      $MPI_NODES"
echo "  SpMV Chunk:     $SPMV_CHUNK"
echo "  Norm Chunk:     $NORM_CHUNK"
echo "=========================================================="

echo "Implementation,Mode,Total_Time_Med,Comp_Time_Med" > "$CSV_OUTPUT"

MODES=("regular" "irregular")

for MODE in "${MODES[@]}"; do
    echo "=========================================================="
    echo ">> Testing Mode: $MODE"
    echo "=========================================================="

    # ---------------------------------------------------------
    # 1. PTHREADS (1 Nodo, 16 Threads)
    # ---------------------------------------------------------
    echo "  -> Running Pthreads (1 Node)..."
    tot_pth=()
    comp_pth=()
    for r in $(seq 1 "$REPEATS"); do
        out=$(srun --time="$SRUN_TIME" -N 1 -n 1 -c "$THREADS" "$PTH_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        tot_pth+=($(echo "$out" | extract_tot_time))
        comp_pth+=($(echo "$out" | extract_comp_time))
    done
    med_tot_pth=$(calculate_median "${tot_pth[@]}")
    med_comp_pth=$(calculate_median "${comp_pth[@]}")
    echo "PTHREADS,$MODE,$med_tot_pth,$med_comp_pth" >> "$CSV_OUTPUT"

    # ---------------------------------------------------------
    # 2. OPENMP (1 Nodo, 16 Threads)
    # ---------------------------------------------------------
    echo "  -> Running OpenMP (1 Node)..."
    tot_omp=()
    comp_omp=()
    for r in $(seq 1 "$REPEATS"); do
        out=$(OMP_NUM_THREADS="$THREADS" srun --time="$SRUN_TIME" -N 1 -n 1 -c "$THREADS" "$OMP_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        tot_omp+=($(echo "$out" | extract_tot_time))
        comp_omp+=($(echo "$out" | extract_comp_time))
    done
    med_tot_omp=$(calculate_median "${tot_omp[@]}")
    med_comp_omp=$(calculate_median "${comp_omp[@]}")
    echo "OPENMP,$MODE,$med_tot_omp,$med_comp_omp" >> "$CSV_OUTPUT"

    # ---------------------------------------------------------
    # 3. MPI + OPENMP (8 Nodi, 1 Task/Nodo, 16 Threads/Task)
    # ---------------------------------------------------------
    echo "  -> Running MPI+OpenMP (8 Nodes)..."
    tot_mpi=()
    comp_mpi=()
    for r in $(seq 1 "$REPEATS"); do
        out=$(OMP_NUM_THREADS="$THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$MPI_NODES" -n "$MPI_NODES" -c "$THREADS" "$MPI_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        tot_mpi+=($(echo "$out" | extract_tot_time))
        comp_mpi+=($(echo "$out" | extract_comp_time))
    done
    med_tot_mpi=$(calculate_median "${tot_mpi[@]}")
    med_comp_mpi=$(calculate_median "${comp_mpi[@]}")
    echo "MPI_OMP,$MODE,$med_tot_mpi,$med_comp_mpi" >> "$CSV_OUTPUT"

done

echo "=========================================================="
echo " Experiment completed!"
echo " Results successfully saved to: $CSV_OUTPUT"
echo "=========================================================="