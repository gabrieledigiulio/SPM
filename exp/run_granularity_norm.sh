#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Vector Ops Granularity Sweep (-nc)
# ==============================================================================

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
CPPTHREADS_BIN="../threadpool_SpMV"
OMP_TASKS_BIN="../omp_tasks_SpMV"
MPI_OMP_BIN="../mpi_omp_tasks_SpMV"

SRUN_TIME="00:10:00"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Problem Parameters
N=1000000
NZ=250000000
MODE="irregular"
SEED=111

# Topologia Hardware (Configurazione vincente a 16 thread)
THREADS=16
MPI_NODES=8
MPI_RANKS=8

# Parametri dello Sweep
SPMV_CHUNK=1024
NORM_CHUNKS=(1024 65536)
REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/norm_granularity_results_16threads.csv"

# ==========================================
# 2. Extraction & Math Functions
# ==========================================
extract_total_time()  { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_vecops_time() { grep -oP 'Vector ops time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time()   { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_epoch_time()  { grep -oP 'Epoch transition \(sec\) = \K[0-9.]+' | head -1 || true; }

extract_comp_time() {
    local vec_time="$1"
    local spmv_time="$2"
    local epoch_time="$3"

    if [[ -z "$vec_time" || -z "$spmv_time" || -z "$epoch_time" || "$vec_time" == "N/A" || "$spmv_time" == "N/A" || "$epoch_time" == "N/A" ]]; then
        echo "N/A"
        return 0
    fi

    awk -v v="$vec_time" -v s="$spmv_time" -v e="$epoch_time" 'BEGIN { printf "%.12f", v + s + e }'
}

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
echo " NORM GRANULARITY SWEEP CONFIGURATION ($REPEATS REPEATS)"
echo "=========================================================="
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Threads per Worker:   $THREADS"
echo "  MPI Topology:         $MPI_NODES Nodes, $MPI_RANKS Ranks"
echo "  SpMV Chunk Size:      $SPMV_CHUNK (Fixed)"
echo "  Norm Chunk Sizes:     ${NORM_CHUNKS[*]}"
echo "=========================================================="

echo "Norm_Chunk,Implementation,Total_Time_Med,Comp_Time_Med,VectorOps_Time_Med,SpMV_Time_Med,Epoch_Time_Med" > "$CSV_OUTPUT"

for nc in "${NORM_CHUNKS[@]}"; do
    echo "=========================================================="
    echo ">> Testing Norm Chunk Size: $nc (Repeats: $REPEATS)"
    echo "=========================================================="

    # Inizializzazione array
    cpp_tot=(); cpp_comp=(); cpp_vec=(); cpp_spmv=(); cpp_ep=()
    omp_tot=(); omp_comp=(); omp_vec=(); omp_spmv=(); omp_ep=()
    mpi_tot=(); mpi_comp=(); mpi_vec=(); mpi_spmv=(); mpi_ep=()

    for r in $(seq 1 "$REPEATS"); do
        echo "  [Run $r/$REPEATS]"
        
        # --- 1. C++ Threads ---
        out_cpp=$(srun --time="$SRUN_TIME" -N 1 -n 1 --cpus-per-task="$THREADS" \
            "$CPPTHREADS_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$nc")
        
        cpp_tot_val=$(echo "$out_cpp" | extract_total_time)
        cpp_vec_val=$(echo "$out_cpp" | extract_vecops_time)
        cpp_spmv_val=$(echo "$out_cpp" | extract_spmv_time)
        cpp_ep_val=$(echo "$out_cpp" | extract_epoch_time)
        cpp_comp_val=$(extract_comp_time "$cpp_vec_val" "$cpp_spmv_val" "$cpp_ep_val")

        cpp_tot+=("$cpp_tot_val")
        cpp_comp+=("$cpp_comp_val")
        cpp_vec+=("$cpp_vec_val")
        cpp_spmv+=("$cpp_spmv_val")
        cpp_ep+=("$cpp_ep_val")

        # --- 2. OpenMP Tasks ---
        out_omp=$(srun --time="$SRUN_TIME" -N 1 -n 1 --cpus-per-task="$THREADS" \
            "$OMP_TASKS_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$nc")
        
        omp_tot_val=$(echo "$out_omp" | extract_total_time)
        omp_vec_val=$(echo "$out_omp" | extract_vecops_time)
        omp_spmv_val=$(echo "$out_omp" | extract_spmv_time)
        omp_ep_val=$(echo "$out_omp" | extract_epoch_time)
        omp_comp_val=$(extract_comp_time "$omp_vec_val" "$omp_spmv_val" "$omp_ep_val")

        omp_tot+=("$omp_tot_val")
        omp_comp+=("$omp_comp_val")
        omp_vec+=("$omp_vec_val")
        omp_spmv+=("$omp_spmv_val")
        omp_ep+=("$omp_ep_val")

        # --- 3. MPI + OpenMP ---
        out_mpi=$(OMP_NUM_THREADS="$THREADS" srun --time="$SRUN_TIME" --mpi=pmix -N "$MPI_NODES" -n "$MPI_RANKS" \
            --cpus-per-task="$THREADS" "$MPI_OMP_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$nc")
        
        mpi_tot_val=$(echo "$out_mpi" | extract_total_time)
        mpi_vec_val=$(echo "$out_mpi" | extract_vecops_time)
        mpi_spmv_val=$(echo "$out_mpi" | extract_spmv_time)
        mpi_ep_val=$(echo "$out_mpi" | extract_epoch_time)
        mpi_comp_val=$(extract_comp_time "$mpi_vec_val" "$mpi_spmv_val" "$mpi_ep_val")

        mpi_tot+=("$mpi_tot_val")
        mpi_comp+=("$mpi_comp_val")
        mpi_vec+=("$mpi_vec_val")
        mpi_spmv+=("$mpi_spmv_val")
        mpi_ep+=("$mpi_ep_val")
    done

    # --- Calcolo mediane e salvataggio su CSV ---
    
    # CPP
    m_cpp_tot=$(calculate_median "${cpp_tot[@]}")
    m_cpp_comp=$(calculate_median "${cpp_comp[@]}")
    m_cpp_vec=$(calculate_median "${cpp_vec[@]}")
    m_cpp_spmv=$(calculate_median "${cpp_spmv[@]}")
    m_cpp_ep=$(calculate_median "${cpp_ep[@]}")
    echo "$nc,CPP_THREADS,$m_cpp_tot,$m_cpp_comp,$m_cpp_vec,$m_cpp_spmv,$m_cpp_ep" >> "$CSV_OUTPUT"
    echo "  -> CPP_THREADS  Medians: Tot=${m_cpp_tot}s, Comp=${m_cpp_comp}s, Vec=${m_cpp_vec}s"

    # OMP
    m_omp_tot=$(calculate_median "${omp_tot[@]}")
    m_omp_comp=$(calculate_median "${omp_comp[@]}")
    m_omp_vec=$(calculate_median "${omp_vec[@]}")
    m_omp_spmv=$(calculate_median "${omp_spmv[@]}")
    m_omp_ep=$(calculate_median "${omp_ep[@]}")
    echo "$nc,OMP_TASKS,$m_omp_tot,$m_omp_comp,$m_omp_vec,$m_omp_spmv,$m_omp_ep" >> "$CSV_OUTPUT"
    echo "  -> OMP_TASKS    Medians: Tot=${m_omp_tot}s, Comp=${m_omp_comp}s, Vec=${m_omp_vec}s"

    # MPI
    m_mpi_tot=$(calculate_median "${mpi_tot[@]}")
    m_mpi_comp=$(calculate_median "${mpi_comp[@]}")
    m_mpi_vec=$(calculate_median "${mpi_vec[@]}")
    m_mpi_spmv=$(calculate_median "${mpi_spmv[@]}")
    m_mpi_ep=$(calculate_median "${mpi_ep[@]}")
    echo "$nc,MPI_OMP,$m_mpi_tot,$m_mpi_comp,$m_mpi_vec,$m_mpi_spmv,$m_mpi_ep" >> "$CSV_OUTPUT"
    echo "  -> MPI_OMP      Medians: Tot=${m_mpi_tot}s, Comp=${m_mpi_comp}s, Vec=${m_mpi_vec}s"

done

echo "=========================================================="
echo " Norm granularity sweep completed!"
echo " Results successfully saved to: $CSV_OUTPUT"
echo "=========================================================="