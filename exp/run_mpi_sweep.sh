#!/usr/bin/env bash
set -euo pipefail

SEP=$(printf '=%.0s' {1..58})

MPI_OMP_BIN="../mpi_omp_tasks_SpMV"
SRUN_TIME="00:10:00"

N=1000000
NZ=250000000
MODE="irregular"
SEED=111
SPMV_CHUNK=2048
NORM_CHUNK=2048
REPEATS=3

MPI_NODES=8
CORES_PER_NODE=16

export OMP_PLACES=cores
export OMP_PROC_BIND=close

RESULT_DIR="results"
CSV_OUTPUT="${RESULT_DIR}/mpi_sweep_results.csv"
mkdir -p "$RESULT_DIR"

echo "Nodes,Total_Ranks,Threads_Per_Rank,Block_Size,Total_Time_Med,Comp_Time_Med,Comm_Time_Med,Red_Time_Med,Epoch_Time_Med,Scatt_Time_Med" > "$CSV_OUTPUT"

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

echo "$SEP"
echo " MPI+OpenMP HYBRID SWEEP ($REPEATS REPEATS)"
echo "$SEP"
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Mode:                 $MODE"
echo "  MPI Nodes:            $MPI_NODES"
echo "  Cores per Node:       $CORES_PER_NODE"
echo "  SpMV Chunk Size:      $SPMV_CHUNK"
echo "  Norm Chunk Size:      $NORM_CHUNK"
echo "$SEP"

for MPI_THREADS in 1 2 4 8 16 32; do

    if [ "$MPI_THREADS" -eq 32 ]; then
        RANKS_PER_NODE=1
        EXTRA_SRUN_ARGS="--oversubscribe"
    else
        RANKS_PER_NODE=$(( CORES_PER_NODE / MPI_THREADS ))
        EXTRA_SRUN_ARGS=""
    fi

    TOTAL_RANKS=$(( RANKS_PER_NODE * MPI_NODES ))

    echo "$SEP"
    echo ">> Ranks=$TOTAL_RANKS ($RANKS_PER_NODE/node), MPI_THREADS=$MPI_THREADS"
    echo "$SEP"

    tot_times=()
    comp_times=()
    comm_times=()
    red_times=()
    epoch_times=()
    scatt_times=()

    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"

        OUTPUT_MPI=$(OMP_NUM_THREADS="$MPI_THREADS" \
            srun --time="$SRUN_TIME" --mpi=pmix \
                 -N "$MPI_NODES" -n "$TOTAL_RANKS" --cpus-per-task="$MPI_THREADS" $EXTRA_SRUN_ARGS \
                 "$MPI_OMP_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$MPI_THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")

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

    echo "  >> MPI_OMP      Medians: Tot=${med_tot}s, Comp=${med_comp}s, Comm=${med_comm}s"

    echo "$MPI_NODES,$TOTAL_RANKS,$MPI_THREADS,$SPMV_CHUNK,$med_tot,$med_comp,$med_comm,$med_red,$med_epoch,$med_scatt" >> "$CSV_OUTPUT"

done

echo "$SEP"
echo " Results saved to: $CSV_OUTPUT"
echo "$SEP"