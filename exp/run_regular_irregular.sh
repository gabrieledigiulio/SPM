#!/usr/bin/env bash
set -euo pipefail

SEP=$(printf '=%.0s' {1..58})

CPPTHREADS_BIN="../threadpool_SpMV"
OMP_TASKS_BIN="../omp_tasks_SpMV"
MPI_OMP_BIN="../mpi_omp_tasks_SpMV"
SRUN_TIME="00:15:00"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

N=1000000
NZ=250000000
SEED=111

CPP_THREADS=16
OMP_THREADS=16
MPI_NODES=8
MPI_RANKS=8
MPI_THREADS=16

SPMV_CHUNK=1024
NORM_CHUNK=16384
REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/regular_vs_irregular.csv"

extract_total_time()  { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_vecops_time() { grep -oP 'Vector ops time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time()   { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_epoch_time()  { grep -oP 'Epoch transition \(sec\) = \K[0-9.]+' | head -1 || true; }

extract_comp_time() {
    local output="$1"

    local explicit_comp=$(echo "$output" | grep -oP 'Computation time \(sec\) = \K[0-9.]+' | head -1 || true)

    if [[ -n "$explicit_comp" && "$explicit_comp" != "N/A" ]]; then
        echo "$explicit_comp"
        return 0
    fi

    local vec_time=$(echo "$output" | extract_vecops_time)
    local spmv_time=$(echo "$output" | extract_spmv_time)
    local epoch_time=$(echo "$output" | extract_epoch_time)

    if [[ -z "$vec_time" || -z "$spmv_time" || -z "$epoch_time" ]]; then
        echo "N/A"
        return 0
    fi

    awk -v v="$vec_time" -v s="$spmv_time" -v e="$epoch_time" 'BEGIN { printf "%.6f", v + s + e }'
}

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
echo " REGULAR vs IRREGULAR ($REPEATS REPEATS)"
echo "$SEP"
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  CPP Threads:          $CPP_THREADS"
echo "  OMP Threads:          $OMP_THREADS"
echo "  MPI Threads:          $MPI_THREADS"
echo "  MPI Topology:         $MPI_NODES Nodes, $MPI_RANKS Ranks"
echo "  SpMV Chunk Size:      $SPMV_CHUNK"
echo "  Norm Chunk Size:      $NORM_CHUNK"
echo "$SEP"

echo "Mode,Implementation,Total_Time_Med,Comp_Time_Med,SpMV_Time_Med" > "$CSV_OUTPUT"

MODES=("regular" "irregular")

for MODE in "${MODES[@]}"; do
    echo "$SEP"
    echo ">> Mode=$MODE"
    echo "$SEP"

    tot_pth=(); comp_pth=(); spmv_pth=();
    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"
        out=$(srun --time="$SRUN_TIME" -N 1 -n 1 -c "$CPP_THREADS" "$CPPTHREADS_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$CPP_THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        tot_pth+=($(echo "$out" | extract_total_time))
        comp_pth+=($(extract_comp_time "$out"))
        spmv_pth+=($(echo "$out" | extract_spmv_time))
    done
    m_tot_pth=$(calculate_median "${tot_pth[@]}")
    m_comp_pth=$(calculate_median "${comp_pth[@]}")
    m_spmv_pth=$(calculate_median "${spmv_pth[@]}")
    echo "$MODE,CPP_THREADS,$m_tot_pth,$m_comp_pth,$m_spmv_pth" >> "$CSV_OUTPUT"
    echo "  >> CPP_THREADS  Medians: Tot=${m_tot_pth}s, Comp=${m_comp_pth}s"

    tot_omp=(); comp_omp=(); spmv_omp=();
    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"
        out=$(OMP_NUM_THREADS="$OMP_THREADS" srun --time="$SRUN_TIME" -N 1 -n 1 -c "$OMP_THREADS" "$OMP_TASKS_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$OMP_THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        tot_omp+=($(echo "$out" | extract_total_time))
        comp_omp+=($(extract_comp_time "$out"))
        spmv_omp+=($(echo "$out" | extract_spmv_time))
    done
    m_tot_omp=$(calculate_median "${tot_omp[@]}")
    m_comp_omp=$(calculate_median "${comp_omp[@]}")
    m_spmv_omp=$(calculate_median "${spmv_omp[@]}")
    echo "$MODE,OMP_TASKS,$m_tot_omp,$m_comp_omp,$m_spmv_omp" >> "$CSV_OUTPUT"
    echo "  >> OMP_TASKS    Medians: Tot=${m_tot_omp}s, Comp=${m_comp_omp}s"

    tot_mpi=(); comp_mpi=(); spmv_mpi=();
    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"
        out=$(OMP_NUM_THREADS="$MPI_THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$MPI_NODES" -n "$MPI_RANKS" -c "$MPI_THREADS" "$MPI_OMP_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$MPI_THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        tot_mpi+=($(echo "$out" | extract_total_time))
        comp_mpi+=($(extract_comp_time "$out"))
        spmv_mpi+=($(echo "$out" | extract_spmv_time))
    done
    m_tot_mpi=$(calculate_median "${tot_mpi[@]}")
    m_comp_mpi=$(calculate_median "${comp_mpi[@]}")
    m_spmv_mpi=$(calculate_median "${spmv_mpi[@]}")
    echo "$MODE,MPI_OMP,$m_tot_mpi,$m_comp_mpi,$m_spmv_mpi" >> "$CSV_OUTPUT"
    echo "  >> MPI_OMP      Medians: Tot=${m_tot_mpi}s, Comp=${m_comp_mpi}s"

done

echo "$SEP"
echo " Results saved to: $CSV_OUTPUT"
echo "$SEP"