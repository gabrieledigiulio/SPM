#!/usr/bin/env bash
set -euo pipefail

SEP=$(printf '=%.0s' {1..58})

MPI_OMP_BIN="../mpi_omp_tasks_SpMV"
SRUN_TIME="00:15:00"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

N=1000000
NZ=250000000
MODE="irregular"
SEED=111

MPI_THREADS=16
SPMV_CHUNK=1024
NORM_CHUNK=16384

NODES_LIST=(1 2 4 8)
REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/strong_scaling_results.csv"

extract_tot_time()   { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comp_time()  { grep -oP 'Computation time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comm_time()  { grep -oP 'Communication time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_red_time()   { grep -oP 'Reduction time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_scatt_time() { grep -oP 'Scatter time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_epoch_time() { grep -oP 'Epoch transition \(sec\) = \K[0-9.]+' | head -1 || true; }

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

echo "$SEP"
echo " STRONG SCALABILITY ($REPEATS REPEATS)"
echo "$SEP"
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Mode:                 $MODE"
echo "  MPI Threads:          $MPI_THREADS"
echo "  SpMV Chunk Size:      $SPMV_CHUNK"
echo "  Norm Chunk Size:      $NORM_CHUNK"
echo "  Nodes:                ${NODES_LIST[*]}"
echo "$SEP"

echo "Nodes,Total_Time_Med,Comp_Time_Med,Comm_Time_Med,Red_Time_Med,Scatt_Time_Med,Epoch_Time_Med" > "$CSV_OUTPUT"

for nodes in "${NODES_LIST[@]}"; do
    echo "$SEP"
    echo ">> Nodes=$nodes, MPI Ranks=$nodes"
    echo "$SEP"

    tot_times=()
    comp_times=()
    comm_times=()
    red_times=()
    scatt_times=()
    epoch_times=()

    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"

        out_mpi=$(OMP_NUM_THREADS="$MPI_THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$nodes" -n "$nodes" \
            --cpus-per-task="$MPI_THREADS" "$MPI_OMP_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" \
            -t "$MPI_THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")

        tot_times+=($(echo "$out_mpi" | extract_tot_time))
        comp_times+=($(echo "$out_mpi" | extract_comp_time))
        comm_times+=($(echo "$out_mpi" | extract_comm_time))
        red_times+=($(echo "$out_mpi" | extract_red_time))
        scatt_times+=($(echo "$out_mpi" | extract_scatt_time))
        epoch_times+=($(echo "$out_mpi" | extract_epoch_time))
    done

    m_tot=$(calculate_median "${tot_times[@]}")
    m_comp=$(calculate_median "${comp_times[@]}")
    m_comm=$(calculate_median "${comm_times[@]}")
    m_red=$(calculate_median "${red_times[@]}")
    m_scatt=$(calculate_median "${scatt_times[@]}")
    m_epoch=$(calculate_median "${epoch_times[@]}")

    echo "$nodes,$m_tot,$m_comp,$m_comm,$m_red,$m_scatt,$m_epoch" >> "$CSV_OUTPUT"
    echo "  -> MPI_OMP      Medians: Tot=${m_tot}s, Comp=${m_comp}s, Comm=${m_comm}s"
done

echo "$SEP"
echo " Completed! Results saved to: $CSV_OUTPUT"
echo "$SEP"