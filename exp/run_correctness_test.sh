#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Correctness & Timing Verification Script (Optimized)
# ==============================================================================

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
SEQ_BIN="../iterative_SpMV"

# Problem Parameters (Small size for fast testing)
N=1000000
NZ=20000000
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

# Numerical tolerance
TOLERANCE="1e-12"

# CSV output
CSV_FILE="results.csv"
CSV_HEADER="label,kind,computation_time,vector_ops_time,spmv_time,scatter_time,communication_time,reduction_time,epoch_transition_time,imbalance,checksum,rayleigh"
printf '%s\n' "$CSV_HEADER" > "$CSV_FILE"

# Vector dump control
ENABLE_DUMP=false
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

write_csv_result() {
    local label="$1"
    local kind="$2"

    local time_var="${label}_TIME"
    local vecops_var="${label}_VECOPS"
    local spmv_var="${label}_SPMV"
    local scatt_var="${label}_SCATT"
    local comm_var="${label}_COMM"
    local red_var="${label}_RED"
    local epoch_var="${label}_EPOCH"
    local imb_var="${label}_IMBAL"
    local chk_var="${label}_CHK"
    local ray_var="${label}_RAY"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$label" \
        "$kind" \
        "${!time_var:-N/A}" \
        "${!vecops_var:-N/A}" \
        "${!spmv_var:-N/A}" \
        "${!scatt_var:-N/A}" \
        "${!comm_var:-N/A}" \
        "${!red_var:-N/A}" \
        "${!epoch_var:-N/A}" \
        "${!imb_var:-N/A}" \
        "${!chk_var:-N/A}" \
        "${!ray_var:-N/A}" >> "$CSV_FILE"
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
    declare -g "${label}_COMP=${comp_time}"
    declare -g "${label}_VECOPS=${vecops_time}"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"
    declare -g "${label}_SPMV=${spmv_time}"
    declare -g "${label}_SCATT=${scatt_time}"
    declare -g "${label}_COMM=${comm_time}"
    declare -g "${label}_RED=${red_time}"
    declare -g "${label}_EPOCH=${epoch_time}"
    declare -g "${label}_IMBAL=${imbalance}"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Computation: ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
    print_optional_metric "Vector ops" "$vecops_time"
    print_optional_metric "SpMV" "$spmv_time"
    print_optional_metric "Scatter" "$scatt_time"
    print_optional_metric "Communication" "$comm_time"
    print_optional_metric "Reduction" "$red_time"
    print_optional_metric "Epoch transition" "$epoch_time"
    print_optional_metric "Imbalance" "$imbalance"

    write_csv_result "$label" "single_node"
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
    declare -g "${label}_COMP=${comp_time}"
    declare -g "${label}_VECOPS=${vecops_time}"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"
    declare -g "${label}_SPMV=${spmv_time}"
    declare -g "${label}_SCATT=${scatt_time}"
    declare -g "${label}_COMM=${comm_time}"
    declare -g "${label}_RED=${red_time}"
    declare -g "${label}_EPOCH=${epoch_time}"
    declare -g "${label}_IMBAL=${imbalance}"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Computation: ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
    print_optional_metric "Vector ops" "$vecops_time"
    print_optional_metric "SpMV" "$spmv_time"
    print_optional_metric "Scatter" "$scatt_time"
    print_optional_metric "Communication" "$comm_time"
    print_optional_metric "Reduction" "$red_time"
    print_optional_metric "Epoch transition" "$epoch_time"
    print_optional_metric "Imbalance" "$imbalance"

    write_csv_result "$label" "mpi_omp"
}

compare_to_seq() {
    local label="$1"
    local dump_file="$2"

    local seq_chk_var="SEQ_CHK" seq_ray_var="SEQ_RAY"
    local par_chk_var="${label}_CHK" par_ray_var="${label}_RAY"

    echo "--- $label vs SEQ ---"

    if [ "${!seq_chk_var}" == "${!par_chk_var}" ]; then
        echo "  [INFO] Checksums Match (${!seq_chk_var})"
    else
        echo "  [INFO] Checksums Differ"
    fi

    local diff_val check_result
    diff_val=$(awk -v s="${!seq_ray_var}" -v p="${!par_ray_var}" \
        'BEGIN { d = s - p; if (d < 0) d = -d; printf "%.2e", d }')
    check_result=$(awk -v s="${!seq_ray_var}" -v p="${!par_ray_var}" -v tol="$TOLERANCE" \
        'BEGIN { d = s - p; if (d < 0) d = -d; print (d <= tol) ? "PASS" : "FAIL" }')

    if [ "$check_result" == "PASS" ]; then
        echo "  [OK] Rayleigh value: VALIDATED (Diff: $diff_val <= $TOLERANCE)"
    else
        echo "  [ERROR] Rayleigh value: OUT OF TOLERANCE! (Diff: $diff_val > $TOLERANCE)"
    fi
    echo ""
}

# ==========================================
# 4. Configuration Summary
# ==========================================
echo "=========================================================="
echo " TEST CONFIGURATION (OPTIMAL REPORT PARAMETERS)"
echo "=========================================================="
echo "  Matrix (N x N):         $N"
echo "  Non-zero elements (NZ): $NZ"
echo "  Matrix mode:            $MODE"
echo "  MPI nodes:              $MPI_NODES"
echo "=========================================================="

# ==========================================
# 5. Sequential Execution
# ==========================================
echo "Sequential Execution"
run_and_extract "SEQ" "$SEQ_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED"
echo "----------------------------------------------------------"

# ==========================================
# 6. Single-node Implementations Execution
# ==========================================
for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    echo "Running $label (-t $th, -c $chk)..."
    run_and_extract "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -t "$th" -c "$chk" -nc "$nchk" -s "$SEED"
    echo "----------------------------------------------------------"
done

# ==========================================
# 7. MPI+OpenMP Implementations Execution
# ==========================================
for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    echo "Running $label (Nodes: $MPI_NODES, MPI_Ranks: $MPI_RANKS, OMP_Threads: $OMP_THREADS_PER_RANK, -c $chk)..."
    run_and_extract_mpi "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -c "$chk" -nc "$nchk" -s "$SEED"
    echo "----------------------------------------------------------"
done

# ==========================================
# 8. Correctness & Timing Verification
# ==========================================
echo "=========================================================="
echo " CORRECTNESS RESULTS (Tolerance: $TOLERANCE)"
echo "=========================================================="

for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    compare_to_seq "$label" "$dump_file"
done
for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    compare_to_seq "$label" "$dump_file"
done

echo "=========================================================="
echo " OVERALL TIMING SUMMARY"
echo "=========================================================="
printf "  %-15s %10s s\n" "SEQ" "${SEQ_TIME:-N/A}"
for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    t_var="${label}_TIME"
    printf "  %-15s %10s s\n" "$label" "${!t_var:-N/A}"
done
for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    t_var="${label}_TIME"
    printf "  %-15s %10s s\n" "$label" "${!t_var:-N/A}"
done
echo "=========================================================="