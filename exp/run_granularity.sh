#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Cross-Validation & Timing Script
# ==============================================================================

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
SEQ_BIN="../iterative_SpMV"

# Problem Parameters (Small size for fast testing & dumping)
N=1000
NZ=25000
MODE="irregular"
SEED=111

# --- Optimal Parameters for C++ Threads ---
CPP_THREADS=32
CPP_CHUNK=2048
CPP_NORM_CHUNK=2048

# --- Optimal Parameters for OpenMP Tasks ---
OMP_THREADS=32
OMP_CHUNK=4096
OMP_NORM_CHUNK=4096

# --- Optimal Parameters for MPI + OpenMP ---
MPI_NODES=8              # Number of physical machines
MPI_RANKS=8              # 1 MPI process per node
OMP_THREADS_PER_RANK=16  # Threads per rank
MPI_CHUNK=1024
MPI_NORM_CHUNK=1024

# Automatic mapping computation for OpenMPI
RANKS_PER_NODE=$(( MPI_RANKS / MPI_NODES ))
MPIRUN_EXTRA_ARGS="--map-by ppr:${RANKS_PER_NODE}:node"

# Numerical tolerance per Rayleigh
TOLERANCE="1e-12"

# Vector dump control (Impostare a 'true' per matrici piccole)
ENABLE_DUMP=true
SEQ_DUMP_FILE="seq_vec.dump"

# ==========================================
# 2. Implementations to Test
# ==========================================
# Format: "LABEL|BINARY|DUMP_FILE|THREADS|CHUNK_SIZE|NORM_CHUNK"
IMPLS=(
    "CPP_THREADS|../threadpool_SpMV|thr_vec.dump|$CPP_THREADS|$CPP_CHUNK|$CPP_NORM_CHUNK"
    "OMP_TASKS|../omp_tasks_SpMV|omp_vec.dump|$OMP_THREADS|$OMP_CHUNK|$OMP_NORM_CHUNK"
)

MPI_IMPLS=(
    "MPI_OMP|../mpi_omp_SpMV|mpi_vec.dump|$OMP_THREADS_PER_RANK|$MPI_CHUNK|$MPI_NORM_CHUNK"
)

# Array per tracciare tutte le run eseguite per il cross-check finale
ALL_LABELS=("SEQ")
ALL_DUMPS=("$SEQ_DUMP_FILE")

# ==========================================
# 3. Extraction & Execution Functions
# ==========================================
extract_comp_time() { grep -oP '(?:Computation time|Time) \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_vecops_time() { grep -oP 'Vector ops time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time() { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_scatt_time(){ grep -oP 'Scatter time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_comm_time() { grep -oP 'Communication time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_red_time()  { grep -oP 'Reduction time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_epoch_time(){ grep -oP 'Epoch transition \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_imbalance() { grep -oP 'imbalance=\K[-0-9.eE+]+' | head -1 || true; }
extract_time()      { extract_comp_time; }
extract_checksum()  { grep -oP 'checksum=\K0x[0-9a-fA-F]+' || true; }
extract_rayleigh()  { grep -oP 'rayleigh=\K[-0-9.eE+]+' || true; }

print_optional_metric() {
    local label="$1" value="$2"
    if [ -n "$value" ]; then
        echo "  -> ${label}: ${value} s"
    fi
}

run_and_extract() {
    local label="$1"; shift
    local out
    out=$("$@")

    local comp_time vecops_time spmv_time scatt_time comm_time red_time epoch_time imbalance
    comp_time=$(echo "$out" | extract_comp_time)
    if [ -z "$comp_time" ]; then
        comp_time=$(echo "$out" | extract_time)
    fi
    vecops_time=$(echo "$out" | extract_vecops_time)
    spmv_time=$(echo "$out" | extract_spmv_time)
    scatt_time=$(echo "$out" | extract_scatt_time)
    comm_time=$(echo "$out" | extract_comm_time)
    red_time=$(echo "$out" | extract_red_time)
    epoch_time=$(echo "$out" | extract_epoch_time)
    imbalance=$(echo "$out" | extract_imbalance)

    declare -g "${label}_TIME=${comp_time}"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Computation: ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
    print_optional_metric "Vector ops" "$vecops_time"
    print_optional_metric "SpMV" "$spmv_time"
    print_optional_metric "Epoch transition" "$epoch_time"
}

run_and_extract_mpi() {
    local label="$1" binary="$2"; shift 2
    local out

    out=$(OMP_NUM_THREADS="$OMP_THREADS_PER_RANK" \
          mpirun -np "$MPI_RANKS" $MPIRUN_EXTRA_ARGS "$binary" "$@")

    local comp_time vecops_time spmv_time scatt_time comm_time red_time epoch_time imbalance
    comp_time=$(echo "$out" | extract_comp_time)
    if [ -z "$comp_time" ]; then
        comp_time=$(echo "$out" | extract_time)
    fi
    vecops_time=$(echo "$out" | extract_vecops_time)
    spmv_time=$(echo "$out" | extract_spmv_time)
    scatt_time=$(echo "$out" | extract_scatt_time)
    comm_time=$(echo "$out" | extract_comm_time)
    red_time=$(echo "$out" | extract_red_time)
    epoch_time=$(echo "$out" | extract_epoch_time)
    imbalance=$(echo "$out" | extract_imbalance)

    declare -g "${label}_TIME=${comp_time}"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Computation: ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
    print_optional_metric "SpMV" "$spmv_time"
    print_optional_metric "Communication" "$comm_time"
    print_optional_metric "Imbalance" "$imbalance"
}

compare_pairs() {
    local l1="$1" d1="$2" l2="$3" d2="$4"
    
    local chk1_var="${l1}_CHK" chk2_var="${l2}_CHK"
    local ray1_var="${l1}_RAY" ray2_var="${l2}_RAY"

    echo "--- $l1 vs $l2 ---"

    # 1. Checksum Comparison
    if [ "${!chk1_var}" == "${!chk2_var}" ]; then
        echo "  [OK] Checksums: MATCH (${!chk1_var})"
    else
        echo "  [INFO] Checksums: DIFFER (${!chk1_var} vs ${!chk2_var})"
    fi

    # 2. Rayleigh Comparison (with Tolerance)
    local diff_val check_result
    diff_val=$(awk -v v1="${!ray1_var}" -v v2="${!ray2_var}" \
        'BEGIN { d = v1 - v2; if (d < 0) d = -d; printf "%.2e", d }')
    check_result=$(awk -v v1="${!ray1_var}" -v v2="${!ray2_var}" -v tol="$TOLERANCE" \
        'BEGIN { d = v1 - v2; if (d < 0) d = -d; print (d <= tol) ? "PASS" : "FAIL" }')

    if [ "$check_result" == "PASS" ]; then
        echo "  [OK] Rayleigh: VALIDATED (Diff: $diff_val <= $TOLERANCE)"
    else
        echo "  [ERROR] Rayleigh: OUT OF TOLERANCE! (Diff: $diff_val > $TOLERANCE)"
    fi

    # 3. Vector Dump Comparison
    if [ "$ENABLE_DUMP" = true ]; then
        if [ -f "$d1" ] && [ -f "$d2" ]; then
            if cmp -s "$d1" "$d2"; then
                echo "  [OK] Dump Files: IDENTICAL"
            else
                echo "  [INFO] Dump Files: DIFFERENT (Expected in parallel contexts)"
            fi
        else
            echo "  [WARN] Dump Files: NOT FOUND for comparison"
        fi
    fi
    echo ""
}

# ==========================================
# 4. Configuration Summary
# ==========================================
echo "=========================================================="
echo " CROSS-VALIDATION CONFIGURATION "
echo "=========================================================="
echo "  Matrix (N x NZ):        $N x $NZ"
echo "  Matrix mode:            $MODE (Seed: $SEED)"
echo "  Enable Dump:            $ENABLE_DUMP"
echo "  Numerical Tolerance:    $TOLERANCE"
echo "=========================================================="

# Costruzione del flag di dump condizionale
DUMP_FLAG=()
if [ "$ENABLE_DUMP" = true ]; then
    DUMP_FLAG=("--dump-vector" "$SEQ_DUMP_FILE")
fi

# ==========================================
# 5. Sequential Execution
# ==========================================
echo "Running SEQ..."
run_and_extract "SEQ" "$SEQ_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" "${DUMP_FLAG[@]}"
echo "----------------------------------------------------------"

# ==========================================
# 6. Single-node Implementations Execution
# ==========================================
for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    ALL_LABELS+=("$label")
    ALL_DUMPS+=("$dump_file")
    
    DUMP_FLAG=()
    if [ "$ENABLE_DUMP" = true ]; then DUMP_FLAG=("--dump-vector" "$dump_file"); fi

    echo "Running $label (-t $th, -c $chk)..."
    run_and_extract "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -t "$th" -c "$chk" -nc "$nchk" -s "$SEED" "${DUMP_FLAG[@]}"
    echo "----------------------------------------------------------"
done

# ==========================================
# 7. MPI+OpenMP Implementations Execution
# ==========================================
for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    ALL_LABELS+=("$label")
    ALL_DUMPS+=("$dump_file")

    DUMP_FLAG=()
    if [ "$ENABLE_DUMP" = true ]; then DUMP_FLAG=("--dump-vector" "$dump_file"); fi

    echo "Running $label (MPI Ranks: $MPI_RANKS, OMP Threads: $OMP_THREADS_PER_RANK)..."
    run_and_extract_mpi "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -c "$chk" -nc "$nchk" -s "$SEED" "${DUMP_FLAG[@]}"
    echo "----------------------------------------------------------"
done

# ==========================================
# 8. Cross-Validation (Tutti contro Tutti)
# ==========================================
echo "=========================================================="
echo " CROSS-VALIDATION RESULTS"
echo "=========================================================="

# Ciclo combinatorio: confronta ogni implementazione con tutte le successive
for (( i=0; i<${#ALL_LABELS[@]}; i++ )); do
    for (( j=i+1; j<${#ALL_LABELS[@]}; j++ )); do
        compare_pairs "${ALL_LABELS[i]}" "${ALL_DUMPS[i]}" "${ALL_LABELS[j]}" "${ALL_DUMPS[j]}"
    done
done

echo "=========================================================="
echo " OVERALL TIMING SUMMARY"
echo "=========================================================="
for (( i=0; i<${#ALL_LABELS[@]}; i++ )); do
    label="${ALL_LABELS[i]}"
    t_var="${label}_TIME"
    printf "  %-15s %10s s\n" "$label" "${!t_var:-N/A}"
done
echo "=========================================================="