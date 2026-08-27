#!/usr/bin/env bash
set -euo pipefail

SEP=$(printf '=%.0s' {1..58})

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
OMP_TASKS_BIN="../omp_tasks_SpMV"
OMP_WORK_BIN="../omp_worksharing_SpMV"
SRUN_TIME="00:15:00"

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

# Variabile dipendente: scaling dei thread all'interno del nodo
THREAD_LIST=(1 2 4 8 16)
REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/omp_task_vs_work.csv"

# ==========================================
# 2. Extraction & Math Functions
# ==========================================
extract_tot_time()  { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comp_time() { grep -oP 'Computation time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time() { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }

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
echo " OMP TASKS vs OMP WORK-SHARING ($REPEATS REPEATS)"
echo "$SEP"
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Mode:                 $MODE"
echo "  SpMV Chunk Size:      $SPMV_CHUNK"
echo "  Norm Chunk Size:      $NORM_CHUNK"
echo "  Threads to test:      ${THREAD_LIST[*]}"
echo "$SEP"

echo "Threads,Implementation,Total_Time_Med,Comp_Time_Med,SpMV_Time_Med" > "$CSV_OUTPUT"

for THREADS in "${THREAD_LIST[@]}"; do
    echo "$SEP"
    echo ">> Testing with Threads: $THREADS"
    echo "$SEP"

    # ---------------------------------------------------------
    # 1. OPENMP TASKS
    # ---------------------------------------------------------
    echo "  -> Running OpenMP Tasks..."
    tot_omp_tasks=()
    comp_omp_tasks=()
    spmv_omp_tasks=()
    
    for r in $(seq 1 "$REPEATS"); do
        out=$(OMP_NUM_THREADS="$THREADS" srun --time="$SRUN_TIME" -N 1 -n 1 -c "$THREADS" "$OMP_TASKS_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        
        tot_omp_tasks+=($(echo "$out" | extract_tot_time))
        comp_omp_tasks+=($(echo "$out" | extract_comp_time))
        spmv_omp_tasks+=($(echo "$out" | extract_spmv_time))
    done
    
    m_tot_tasks=$(calculate_median "${tot_omp_tasks[@]}")
    m_comp_tasks=$(calculate_median "${comp_omp_tasks[@]}")
    m_spmv_tasks=$(calculate_median "${spmv_omp_tasks[@]}")
    
    echo "$THREADS,OMP_TASKS,$m_tot_tasks,$m_comp_tasks,$m_spmv_tasks" >> "$CSV_OUTPUT"
    echo "     Medians [Tasks]: Tot=${m_tot_tasks}s, Comp=${m_comp_tasks}s"

    # ---------------------------------------------------------
    # 2. OPENMP WORK-SHARING
    # ---------------------------------------------------------
    echo "  -> Running OpenMP Work-Sharing (omp for)..."
    tot_omp_work=()
    comp_omp_work=()
    spmv_omp_work=()
    
    for r in $(seq 1 "$REPEATS"); do
        out=$(OMP_NUM_THREADS="$THREADS" srun --time="$SRUN_TIME" -N 1 -n 1 -c "$THREADS" "$OMP_WORK_BIN" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -t "$THREADS" -c "$SPMV_CHUNK" -nc "$NORM_CHUNK")
        
        tot_omp_work+=($(echo "$out" | extract_tot_time))
        comp_omp_work+=($(echo "$out" | extract_comp_time))
        spmv_omp_work+=($(echo "$out" | extract_spmv_time))
    done
    
    m_tot_work=$(calculate_median "${tot_omp_work[@]}")
    m_comp_work=$(calculate_median "${comp_omp_work[@]}")
    m_spmv_work=$(calculate_median "${spmv_omp_work[@]}")
    
    echo "$THREADS,OMP_WORKSHARING,$m_tot_work,$m_comp_work,$m_spmv_work" >> "$CSV_OUTPUT"
    echo "     Medians [Work ]: Tot=${m_tot_work}s, Comp=${m_comp_work}s"

done

echo "$SEP"
echo " Experiment completed! Results saved to: $CSV_OUTPUT"
echo "$SEP"