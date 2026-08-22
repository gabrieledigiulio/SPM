#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Sequential Baseline Evaluation
# ==============================================================================

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
SEQ_BIN="../sequential_SpMV"

# Tempo generoso: 1 core impiegherà molto di più rispetto a 16 core!
SRUN_TIME="00:30:00" 

# Problem Parameters (Fissi per tutto il progetto)
N=1000000
NZ=250000000
MODE="irregular"
SEED=111

REPEATS=1

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/sequential_results.csv"

# ==========================================
# 2. Extraction & Math Functions
# ==========================================
extract_tot_time() { grep -oP '^Time \(sec\) = \K[0-9.]+' | head -1 || true; }

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
echo " SEQUENTIAL BASELINE EXPERIMENT ($REPEATS REPEATS)"
echo "=========================================================="
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Mode:                 $MODE"
echo "  Seed:                 $SEED"
echo "=========================================================="

echo "Implementation,Total_Time_Med" > "$CSV_OUTPUT"

tot_times=()

for r in $(seq 1 "$REPEATS"); do
    echo "  [Run $r/$REPEATS] (This might take a while...)"
    
    # Esecuzione strettamente su 1 nodo, 1 task, 1 core
    out_seq=$(srun --time="$SRUN_TIME" -N 1 -n 1 -c 1 \
        "$SEQ_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED")
    
    run_time=$(echo "$out_seq" | extract_tot_time)
    tot_times+=("$run_time")
    
    echo "    -> Completed in: ${run_time}s"
done

# --- Calcolo mediana ---
m_tot=$(calculate_median "${tot_times[@]}")

# Salvataggio su CSV
echo "SEQUENTIAL,$m_tot" >> "$CSV_OUTPUT"

echo "=========================================================="
echo " Sequential evaluation completed!"
echo " Median Total Time: ${m_tot}s"
echo " Results successfully saved to: $CSV_OUTPUT"
echo "=========================================================="