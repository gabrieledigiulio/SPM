#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Script di verifica correttezza e tempi (Ottimizzato)
# ==============================================================================

# ==========================================
# 1. Configurazione Parametri del Test
# ==========================================
SEQ_BIN="../iterative_SpMV"

# Parametri del problema (Taglia "piccola" per testing veloce)
N=1000
NZ=20000
MODE="irregular"
SEED=111

# --- Parametri Ottimali per C++ Threads ---
CPP_THREADS=32
CPP_CHUNK=2048
CPP_NORM_CHUNK=2048

# --- Parametri Ottimali per OpenMP Tasks ---
OMP_THREADS=32
OMP_CHUNK=4096
OMP_NORM_CHUNK=4096

# --- Parametri Ottimali per MPI + OpenMP ---
MPI_NODES=8              # Numero di macchine fisiche
MPI_RANKS=8              # 1 processo MPI per nodo
OMP_THREADS_PER_RANK=16  # Thread per rank
MPI_CHUNK=1024
MPI_NORM_CHUNK=1024

# Calcolo automatico della mappatura per OpenMPI
RANKS_PER_NODE=$(( MPI_RANKS / MPI_NODES ))
MPIRUN_EXTRA_ARGS="--map-by ppr:${RANKS_PER_NODE}:node"

# Tolleranza numerica
TOLERANCE="1e-12"

# Controllo del dump dei vettori
ENABLE_DUMP=false
SEQ_DUMP_FILE="seq_vec.dump"

# ==========================================
# 2. Implementazioni da testare
# ==========================================
# Formato: "LABEL|BINARIO|FILE_DUMP|THREADS|CHUNK_SIZE|NORM_CHUNK"
IMPLS=(
    "CPP_THREADS|../threadpool_SpMV|thr_vec.dump|$CPP_THREADS|$CPP_CHUNK|$CPP_NORM_CHUNK"
    "OMP_TASKS|../omp_tasks_SpMV|omp_vec.dump|$OMP_THREADS|$OMP_CHUNK|$OMP_NORM_CHUNK"
)

MPI_IMPLS=(
    "MPI_OMP|../mpi_omp_SpMV|mpi_vec.dump|$OMP_THREADS_PER_RANK|$MPI_CHUNK|$MPI_NORM_CHUNK"
)

# ==========================================
# 3. Funzioni di estrazione ed esecuzione
# ==========================================
extract_comp_time() { grep -oP '(?:Computation time|Time) \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time() { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_dot_time()  { grep -oP 'Vector dot time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_scale_time(){ grep -oP 'Vector scale time \(sec\) = \K[0-9.]+' | head -1 || true; }
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

    local comp_time spmv_time dot_time scale_time scatt_time comm_time red_time epoch_time imbalance
    comp_time=$(echo "$out" | extract_comp_time)
    if [ -z "$comp_time" ]; then
        comp_time=$(echo "$out" | extract_time)
    fi
    spmv_time=$(echo "$out" | extract_spmv_time)
    dot_time=$(echo "$out" | extract_dot_time)
    scale_time=$(echo "$out" | extract_scale_time)
    scatt_time=$(echo "$out" | extract_scatt_time)
    comm_time=$(echo "$out" | extract_comm_time)
    red_time=$(echo "$out" | extract_red_time)
    epoch_time=$(echo "$out" | extract_epoch_time)
    imbalance=$(echo "$out" | extract_imbalance)

    declare -g "${label}_TIME=${comp_time}"
    declare -g "${label}_COMP=${comp_time}"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"
    declare -g "${label}_SPMV=${spmv_time}"
    declare -g "${label}_DOT=${dot_time}"
    declare -g "${label}_SCALE=${scale_time}"
    declare -g "${label}_SCATT=${scatt_time}"
    declare -g "${label}_COMM=${comm_time}"
    declare -g "${label}_RED=${red_time}"
    declare -g "${label}_EPOCH=${epoch_time}"
    declare -g "${label}_IMBAL=${imbalance}"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Computation: ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
    print_optional_metric "SpMV" "$spmv_time"
    print_optional_metric "Vector dot" "$dot_time"
    print_optional_metric "Vector scale" "$scale_time"
    print_optional_metric "Scatter" "$scatt_time"
    print_optional_metric "Communication" "$comm_time"
    print_optional_metric "Reduction" "$red_time"
    print_optional_metric "Epoch transition" "$epoch_time"
    print_optional_metric "Imbalance" "$imbalance"
}

run_and_extract_mpi() {
    local label="$1" binary="$2"; shift 2
    local out

    out=$(OMP_NUM_THREADS="$OMP_THREADS_PER_RANK" \
          mpirun -np "$MPI_RANKS" $MPIRUN_EXTRA_ARGS "$binary" "$@")

    local comp_time spmv_time dot_time scale_time scatt_time comm_time red_time epoch_time imbalance
    comp_time=$(echo "$out" | extract_comp_time)
    if [ -z "$comp_time" ]; then
        comp_time=$(echo "$out" | extract_time)
    fi
    spmv_time=$(echo "$out" | extract_spmv_time)
    dot_time=$(echo "$out" | extract_dot_time)
    scale_time=$(echo "$out" | extract_scale_time)
    scatt_time=$(echo "$out" | extract_scatt_time)
    comm_time=$(echo "$out" | extract_comm_time)
    red_time=$(echo "$out" | extract_red_time)
    epoch_time=$(echo "$out" | extract_epoch_time)
    imbalance=$(echo "$out" | extract_imbalance)

    declare -g "${label}_TIME=${comp_time}"
    declare -g "${label}_COMP=${comp_time}"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"
    declare -g "${label}_SPMV=${spmv_time}"
    declare -g "${label}_DOT=${dot_time}"
    declare -g "${label}_SCALE=${scale_time}"
    declare -g "${label}_SCATT=${scatt_time}"
    declare -g "${label}_COMM=${comm_time}"
    declare -g "${label}_RED=${red_time}"
    declare -g "${label}_EPOCH=${epoch_time}"
    declare -g "${label}_IMBAL=${imbalance}"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Computation: ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
    print_optional_metric "SpMV" "$spmv_time"
    print_optional_metric "Vector dot" "$dot_time"
    print_optional_metric "Vector scale" "$scale_time"
    print_optional_metric "Scatter" "$scatt_time"
    print_optional_metric "Communication" "$comm_time"
    print_optional_metric "Reduction" "$red_time"
    print_optional_metric "Epoch transition" "$epoch_time"
    print_optional_metric "Imbalance" "$imbalance"
}

compare_to_seq() {
    local label="$1"
    local dump_file="$2"

    local seq_chk_var="SEQ_CHK" seq_ray_var="SEQ_RAY"
    local par_chk_var="${label}_CHK" par_ray_var="${label}_RAY"

    echo "--- $label vs SEQ ---"

    if [ "${!seq_chk_var}" == "${!par_chk_var}" ]; then
        echo "  [INFO] Checksum Uguali (${!seq_chk_var})"
    else
        echo "  [INFO] Checksum Differenti"
    fi

    local diff_val check_result
    diff_val=$(awk -v s="${!seq_ray_var}" -v p="${!par_ray_var}" \
        'BEGIN { d = s - p; if (d < 0) d = -d; printf "%.2e", d }')
    check_result=$(awk -v s="${!seq_ray_var}" -v p="${!par_ray_var}" -v tol="$TOLERANCE" \
        'BEGIN { d = s - p; if (d < 0) d = -d; print (d <= tol) ? "PASS" : "FAIL" }')

    if [ "$check_result" == "PASS" ]; then
        echo "  [OK] Rayleigh value: VALIDATO (Diff: $diff_val <= $TOLERANCE)"
    else
        echo "  [ERRORE] Rayleigh value: FUORI TOLLERANZA! (Diff: $diff_val > $TOLERANCE)"
    fi
    echo ""
}

# ==========================================
# 4. Riepilogo configurazione
# ==========================================
echo "=========================================================="
echo " CONFIGURAZIONE TEST (PARAMETRI OTTIMALI REPORT)"
echo "=========================================================="
echo "  Matrice (N x N):        $N"
echo "  Elementi non-nulli (NZ): $NZ"
echo "  Modalità matrice:       $MODE"
echo "  MPI nodes:              $MPI_NODES"
echo "=========================================================="

# ==========================================
# 5. Esecuzione Sequenziale
# ==========================================
echo "Esecuzione Sequenziale"
run_and_extract "SEQ" "$SEQ_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED"
echo "----------------------------------------------------------"

# ==========================================
# 6. Esecuzione versioni a singolo nodo
# ==========================================
for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    echo "Esecuzione $label (-t $th, -c $chk)..."
    run_and_extract "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -t "$th" -c "$chk" -nc "$nchk" -s "$SEED"
    echo "----------------------------------------------------------"
done

# ==========================================
# 7. Esecuzione versioni MPI+OpenMP
# ==========================================
for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file th chk nchk <<< "$entry"
    echo "Esecuzione $label (Nodes: $MPI_NODES, MPI_Ranks: $MPI_RANKS, OMP_Threads: $OMP_THREADS_PER_RANK, -c $chk)..."
    run_and_extract_mpi "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -c "$chk" -nc "$nchk" -s "$SEED"
    echo "----------------------------------------------------------"
done

# ==========================================
# 8. Verifica di Correttezza e Tempi
# ==========================================
echo "=========================================================="
echo " RISULTATI CORRETTEZZA (Tolleranza: $TOLERANCE)"
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
echo " RIEPILOGO TEMPI COMPLESSIVI"
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