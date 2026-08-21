#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Granularity Sweep Script (Block Size)
# ==============================================================================

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
CPPTHREADS_BIN="../threadpool_SpMV"
OMP_TASKS_BIN="../omp_tasks_SpMV"
MPI_OMP_BIN="../mpi_omp_tasks_SpMV"

# Tempo massimo per ogni singola chiamata srun
SRUN_TIME="00:10:00"

# Affinity dei thread OpenMP
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Problem Parameters (Usiamo il problema grande per stressare la cache)
N=1000000
NZ=250000000
MODE="irregular"
SEED=111

# Topologia Hardware Fissata (La configurazione vincente dallo sweep MPI)
THREADS=32
MPI_NODES=8
MPI_RANKS=8

# Array dei Block Sizes da testare
BLOCK_SIZES=(256 512 1024 2048 4096 8192 16384)

# Configurazione Output
RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/granularity_results_32threads.csv"

# ==========================================
# 2. Extraction Functions
# ==========================================
extract_comp_time() { grep -oP '(?:Computation time|Time) \(sec\) = \K[0-9.]+' | head -1 || true; }

# ==========================================
# 3. Execution Logic
# ==========================================
echo "=========================================================="
echo " GRANULARITY SWEEP CONFIGURATION "
echo "=========================================================="
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Matrix mode:          $MODE (Seed: $SEED)"
echo "  Threads per Worker:   $THREADS"
echo "  MPI Topology:         $MPI_NODES Nodes, $MPI_RANKS Ranks"
echo "  Block Sizes:          ${BLOCK_SIZES[*]}"
echo "=========================================================="

# Creazione intestazione CSV
echo "Block_Size,Implementation,Computation_Time_s" > "$CSV_OUTPUT"

for bs in "${BLOCK_SIZES[@]}"; do
    echo "=========================================================="
    echo " Testing Block Size: $bs"
    echo "=========================================================="

    # --- 1. C++ Threads ---
    echo -n "  -> Running CPP_THREADS... "
    out_cpp=$(srun --time="$SRUN_TIME" -N 1 -n 1 --cpus-per-task="$THREADS" \
        "$CPPTHREADS_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$bs" -nc "$bs")
    t_cpp=$(echo "$out_cpp" | extract_comp_time)
    echo "${t_cpp:-N/A} s"
    echo "$bs,CPP_THREADS,${t_cpp:-N/A}" >> "$CSV_OUTPUT"

    # --- 2. OpenMP Tasks ---
    echo -n "  -> Running OMP_TASKS... "
    out_omp=$(srun --time="$SRUN_TIME" -N 1 -n 1 --cpus-per-task="$THREADS" \
        "$OMP_TASKS_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$bs" -nc "$bs")
    t_omp=$(echo "$out_omp" | extract_comp_time)
    echo "${t_omp:-N/A} s"
    echo "$bs,OMP_TASKS,${t_omp:-N/A}" >> "$CSV_OUTPUT"

    # --- 3. MPI + OpenMP ---
    echo -n "  -> Running MPI_OMP... "
    out_mpi=$(OMP_NUM_THREADS="$THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$MPI_NODES" -n "$MPI_RANKS" \
        --cpus-per-task="$THREADS" "$MPI_OMP_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$bs" -nc "$bs")
    t_mpi=$(echo "$out_mpi" | extract_comp_time)
    echo "${t_mpi:-N/A} s"
    echo "$bs,MPI_OMP,${t_mpi:-N/A}" >> "$CSV_OUTPUT"

done

echo "=========================================================="
echo " Granularity sweep completed!"
echo " Results successfully saved to: $CSV_OUTPUT"
echo "=========================================================="