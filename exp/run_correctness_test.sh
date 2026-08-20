#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Cross-Validation & Timing Script (Slurm/srun style)
# ==============================================================================
#
# Ogni binario viene lanciato con un `srun` "nudo" (senza sbatch/salloc che lo
# avvolge): ogni chiamata è una richiesta di allocazione a sé stante. Slurm la
# mette in coda se le risorse non sono libere, la esegue quando lo sono, e
# rilascia i nodi subito dopo. Per questo la parte MPI (-N 8) occupa gli 8
# nodi solo per la durata di quella singola run, lasciando tutto il resto del
# tempo libero per altri job sul cluster.
#

# ==========================================
# 1. Test Parameter Configuration
# ==========================================
SEQ_BIN="../iterative_SpMV"
CPPTHREADS_BIN="../threadpool_SpMV"
OMP_TASKS_BIN="../omp_tasks_SpMV"
MPI_OMP_BIN="../mpi_omp_tasks_SpMV"

# Tempo massimo per ogni singola chiamata srun
SRUN_TIME="00:05:00"

# Affinity dei thread OpenMP (coerente su tutte le run)
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Problem Parameters (Small size for fast testing & dumping)
N=1000
NZ=25000
MODE="irregular"
SEED=111

# --- Parametri per C++ Threads ---
CPP_THREADS=16
CPP_CHUNK=16
CPP_NORM_CHUNK=16

# --- Parametri per OpenMP Tasks ---
OMP_THREADS=16
OMP_CHUNK=16
OMP_NORM_CHUNK=16

# --- Parametri per MPI + OpenMP ---
MPI_NODES=8              # Numero di nodi fisici richiesti solo per questa fase
MPI_RANKS=8               # 1 rank MPI per nodo
OMP_THREADS_PER_RANK=16   # Thread OpenMP per rank
MPI_CHUNK=16
MPI_NORM_CHUNK=16

# Numerical tolerance per Rayleigh
TOLERANCE="1e-12"

# Vector dump control
ENABLE_DUMP=true
SEQ_DUMP_FILE="seq_vec.dump"

# ==========================================
# 2. Implementations to Test
# ==========================================
# Format: "LABEL|BINARY|DUMP_FILE|THREADS|CHUNK_SIZE|NORM_CHUNK"
IMPLS=(
    "CPP_THREADS|$CPPTHREADS_BIN|thr_vec.dump|$CPP_THREADS|$CPP_CHUNK|$CPP_NORM_CHUNK"
    "OMP_TASKS|$OMP_TASKS_BIN|omp_vec.dump|$OMP_THREADS|$OMP_CHUNK|$OMP_NORM_CHUNK"
)

MPI_IMPLS=(
    "MPI_OMP|$MPI_OMP_BIN|mpi_vec.dump|$OMP_THREADS_PER_RANK|$MPI_CHUNK|$MPI_NORM_CHUNK"
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
extract_epoch_time(){ grep -oP 'Epoch transition \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_time()      { extract_comp_time; }
extract_checksum()  { grep -oP 'checksum=\K0x[0-9a-fA-F]+' || true; }
extract_rayleigh()  { grep -oP 'rayleigh=\K[-0-9.eE+]+' || true; }

print_optional_metric() {
    local label="$1" value="$2"
    if [ -n "$value" ]; then
        echo "  -> ${label}: ${value} s"
    fi
}

# Esegue un binario locale (1 nodo, 1 processo) con srun: la chiamata resta
# in coda finché Slurm non trova un nodo libero, poi lo rilascia a fine run.
run_and_extract() {
    local label="$1" binary="$2"; shift 2
    local out

    out=$(srun --time="$SRUN_TIME" -N 1 -n 1 "$binary" "$@")

    local comp_time vecops_time spmv_time epoch_time
    comp_time=$(echo "$out" | extract_comp_time)
    if [ -z "$comp_time" ]; then
        comp_time=$(echo "$out" | extract_time)
    fi
    vecops_time=$(echo "$out" | extract_vecops_time)
    spmv_time=$(echo "$out" | extract_spmv_time)
    epoch_time=$(echo "$out" | extract_epoch_time)

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

# Esegue il binario MPI+OpenMP con srun: chiede MPI_NODES nodi (e li tiene
# solo per questa chiamata), 1 rank per nodo, OMP_THREADS_PER_RANK thread
# ciascuno. --mpi=pmix e' il plugin di bootstrap richiesto per lanciare i
# rank MPI sotto Slurm.
run_and_extract_mpi() {
    local label="$1" binary="$2"; shift 2
    local out

    out=$(OMP_NUM_THREADS="$OMP_THREADS_PER_RANK" \
          srun --time="$SRUN_TIME" --mpi=pmix -N "$MPI_NODES" -n "$MPI_RANKS" \
               "$binary" "$@")

    local comp_time
    comp_time=$(echo "$out" | extract_comp_time)
    if [ -z "$comp_time" ]; then
        comp_time=$(echo "$out" | extract_time)
    fi

    declare -g "${label}_TIME=${comp_time}"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Computation: ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
}

compare_pairs() {
    local l1="$1" d1="$2" l2="$3" d2="$4"

    local chk1_var="${l1}_CHK" chk2_var="${l2}_CHK"
    local ray1_var="${l1}_RAY" ray2_var="${l2}_RAY"

    echo "--- $l1 vs $l2 ---"

    # 1. Checksum Comparison
    if [ "${!chk1_var}" == "${!chk2_var}" ] && [ -n "${!chk1_var}" ]; then
        echo "  [OK] Checksums: MATCH (${!chk1_var})"
    else
        echo "  [INFO] Checksums: DIFFER (${!chk1_var:-N/A} vs ${!chk2_var:-N/A})"
    fi

    # 2. Rayleigh Comparison (with Tolerance)
    if [ -n "${!ray1_var}" ] && [ -n "${!ray2_var}" ]; then
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
    else
        echo "  [ERROR] Rayleigh: MISSING VALUES"
    fi

    # 3. Vector Dump Comparison
    if [ "$ENABLE_DUMP" = true ]; then
        if [ -f "$d1" ] && [ -f "$d2" ]; then
            if cmp -s "$d1" "$d2"; then
                echo "  [OK] Dump Files: IDENTICAL"
            else
                echo "  [INFO] Dump Files: DIFFERENT (Expected in parallel contexts without ordered floating ops)"
            fi
        else
            echo "  [WARN] Dump Files: NOT FOUND for comparison"
        fi
    fi
    echo ""
}

# ==========================================
# 4. Execution
# ==========================================
echo "=========================================================="
echo " CROSS-VALIDATION CONFIGURATION "
echo "=========================================================="
echo "  Matrix (N x NZ):        $N x $NZ"
echo "  Matrix mode:            $MODE (Seed: $SEED)"
echo "  Enable Dump:            $ENABLE_DUMP"
echo "  Numerical Tolerance:    $TOLERANCE"
echo "  srun time limit/call:   $SRUN_TIME"
echo "=========================================================="

# Costruzione del flag di dump condizionale
DUMP_FLAG=()
if [ "$ENABLE_DUMP" = true ]; then
    DUMP_FLAG=("--dump-vector" "$SEQ_DUMP_FILE")
fi

echo "Running SEQ..."
run_and_extract "SEQ" "$SEQ_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" "${DUMP_FLAG[@]}"
echo "----------------------------------------------------------"

for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    ALL_LABELS+=("$label")
    ALL_DUMPS+=("$dump_file")

    DUMP_FLAG=()
    if [ "$ENABLE_DUMP" = true ]; then DUMP_FLAG=("--dump-vector" "$dump_file"); fi

    echo "Running $label (-N 1, -t $th, -c $chk, -nc $nchk)..."
    run_and_extract "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -t "$th" -c "$chk" -nc "$nchk" -s "$SEED" "${DUMP_FLAG[@]}"
    echo "----------------------------------------------------------"
done

for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    ALL_LABELS+=("$label")
    ALL_DUMPS+=("$dump_file")

    DUMP_FLAG=()
    if [ "$ENABLE_DUMP" = true ]; then DUMP_FLAG=("--dump-vector" "$dump_file"); fi

    echo "Running $label (-N $MPI_NODES, -n $MPI_RANKS, -t $th, -c $chk, -nc $nchk)..."
    run_and_extract_mpi "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -t "$th" -c "$chk" -nc "$nchk" -s "$SEED" "${DUMP_FLAG[@]}"
    echo "----------------------------------------------------------"
done

# ==========================================
# 5. Cross-Validation (Tutti contro Tutti)
# ==========================================
echo "=========================================================="
echo " CROSS-VALIDATION RESULTS"
echo "=========================================================="

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