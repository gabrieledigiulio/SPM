#!/usr/bin/env bash
set -euo pipefail

SEP=$(printf '=%.0s' {1..58})

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
MPI_TASKS_BIN="../mpi_omp_tasks_SpMV"
MPI_WORK_BIN="../mpi_omp_worksharing_SpMV"
SRUN_TIME="00:30:00"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Problem Parameters (Fissi, stress test su irregular)
N=1000000
NZ=250000000
MODE="irregular"
SEED=111

# Ottimizzazioni
SPMV_CHUNK=1024
NORM_CHUNK=16384

# Thread OpenMP per ciascun rank MPI
MPI_THREADS=16

# Scaling dei nodi (1 rank MPI per nodo -> 1, 2, 4, 8 rank)
NODE_LIST=(1 2 4 8)
REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/mpi_task_vs_work.csv"

# ==========================================
# 2. Extraction & Math Functions
# ==========================================
extract_tot_time()  { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comp_time() { grep -oP 'Computation time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time() { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comm_time() { grep -oP 'Communication time \(sec\) = \K[0-9.]+' | head -1 || true; }

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
echo "$SEP"
echo " MPI+OMP TASKS vs MPI+OMP WORK-SHARING ($REPEATS REPEATS)"
echo "$SEP"
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Mode:                 $MODE"
echo "  SpMV Chunk Size:      $SPMV_CHUNK"
echo "  Norm Chunk Size:      $NORM_CHUNK"
echo "  MPI Threads/Rank:     $MPI_THREADS"
echo "  Nodes to test:        ${NODE_LIST[*]}"
echo "$SEP"

echo "Nodes,Implementation,Total_Time_Med,Comp_Time_Med,SpMV_Time_Med,Comm_Time_Med" > "$CSV_OUTPUT"

for NODES in "${NODE_LIST[@]}"; do
    echo "$SEP"
    echo ">> Testing with Nodes: $NODES (1 rank/node, $MPI_THREADS threads/rank)"
    echo "$SEP"

    # ---------------------------------------------------------
    # 1. MPI + OPENMP TASKS
    # ---------------------------------------------------------
    echo "  -> Running MPI + OpenMP Tasks..."
    tot_mpi_tasks=()
    comp_mpi_tasks=()
    spmv_mpi_tasks=()
    comm_mpi_tasks=()

    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"
        out=$(OMP_NUM_THREADS="$MPI_THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$NODES" -n "$NODES" -c "$MPI_THREADS" "$MPI_TASKS_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$MPI_THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")

        tot_mpi_tasks+=($(echo "$out" | extract_tot_time))
        comp_mpi_tasks+=($(echo "$out" | extract_comp_time))
        spmv_mpi_tasks+=($(echo "$out" | extract_spmv_time))
        comm_mpi_tasks+=($(echo "$out" | extract_comm_time))
    done

    m_tot_tasks=$(calculate_median "${tot_mpi_tasks[@]}")
    m_comp_tasks=$(calculate_median "${comp_mpi_tasks[@]}")
    m_spmv_tasks=$(calculate_median "${spmv_mpi_tasks[@]}")
    m_comm_tasks=$(calculate_median "${comm_mpi_tasks[@]}")

    echo "$NODES,MPI_OMP_TASKS,$m_tot_tasks,$m_comp_tasks,$m_spmv_tasks,$m_comm_tasks" >> "$CSV_OUTPUT"
    echo "     Medians [Tasks]: Tot=${m_tot_tasks}s, Comp=${m_comp_tasks}s, Comm=${m_comm_tasks}s"

    # ---------------------------------------------------------
    # 2. MPI + OPENMP WORK-SHARING
    # ---------------------------------------------------------
    echo "  -> Running MPI + OpenMP Work-Sharing (omp for)..."
    tot_mpi_work=()
    comp_mpi_work=()
    spmv_mpi_work=()
    comm_mpi_work=()

    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"
        out=$(OMP_NUM_THREADS="$MPI_THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$NODES" -n "$NODES" -c "$MPI_THREADS" "$MPI_WORK_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$MPI_THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")

        tot_mpi_work+=($(echo "$out" | extract_tot_time))
        comp_mpi_work+=($(echo "$out" | extract_comp_time))
        spmv_mpi_work+=($(echo "$out" | extract_spmv_time))
        comm_mpi_work+=($(echo "$out" | extract_comm_time))
    done

    m_tot_work=$(calculate_median "${tot_mpi_work[@]}")
    m_comp_work=$(calculate_median "${comp_mpi_work[@]}")
    m_spmv_work=$(calculate_median "${spmv_mpi_work[@]}")
    m_comm_work=$(calculate_median "${comm_mpi_work[@]}")

    echo "$NODES,MPI_OMP_WORKSHARING,$m_tot_work,$m_comp_work,$m_spmv_work,$m_comm_work" >> "$CSV_OUTPUT"
    echo "     Medians [Work ]: Tot=${m_tot_work}s, Comp=${m_comp_work}s, Comm=${m_comm_work}s"

done

echo "$SEP"
echo " Experiment completed! Results saved to: $CSV_OUTPUT"
echo "$SEP"
