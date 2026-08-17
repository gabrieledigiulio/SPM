#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPM "One-Shot" Project - Script di verifica correttezza e tempi
# ==============================================================================

# ==========================================
# 1. Configurazione Parametri del Test
# ==========================================
SEQ_BIN="../iterative_SpMV"

# Parametri del problema
N=500000
NZ=20000000
MODE="irregular"
SEED=111

# Parametri del modello parallelo a singolo nodo (C++ threads e OpenMP tasks)
THREADS=16
CHUNK_SIZE=1000          # Granularità SpMV (-c)
NORM_CHUNK=1000          # Granularità operazioni vettoriali (-nc, 0 = automatico)

# --- Parametri specifici per l'architettura distribuita MPI+OpenMP ---
MPI_NODES=2              # Numero di macchine fisiche (nodi) da utilizzare
MPI_RANKS=4              # Numero TOTALE di processi MPI nel cluster
OMP_THREADS_PER_RANK=4   # Thread OpenMP per ogni singolo processo MPI

# Calcolo automatico della mappatura per OpenMPI
RANKS_PER_NODE=$(( MPI_RANKS / MPI_NODES ))
MPIRUN_EXTRA_ARGS="--map-by ppr:${RANKS_PER_NODE}:node"
# (Opzionale per test locali sovraffollati: aggiungere "--oversubscribe")
# ---------------------------------------------------------------------

# Tolleranza numerica
TOLERANCE="1e-12"

# Controllo del dump dei vettori
ENABLE_DUMP=true
SEQ_DUMP_FILE="seq_vec.dump"

# ==========================================
# 2. Implementazioni da testare
# ==========================================
# Formato: "LABEL|BINARIO|FILE_DUMP"
IMPLS=(
    "CPP_THREADS|../threadpool_SpMV|thr_vec.dump"
    "OMP_TASKS|../omp_tasks_SpMV|omp_vec.dump"
)

MPI_IMPLS=(
    "MPI_OMP|../mpi_omp_SpMV|mpi_vec.dump"
)

# ==========================================
# 3. Funzioni di estrazione ed esecuzione
# ==========================================
extract_time()     { grep -oP 'Time \(sec\) = \K[0-9.]+' | head -1; }
extract_checksum() { grep -oP 'checksum=\K0x[0-9a-fA-F]+'; }
extract_rayleigh() { grep -oP 'rayleigh=\K[-0-9.eE+]+'; }

run_and_extract() {
    local label="$1"; shift
    local out
    out=$("$@")

    declare -g "${label}_TIME=$(echo "$out" | extract_time)"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Tempo:     ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
}

run_and_extract_mpi() {
    local label="$1" binary="$2"; shift 2
    local out

    # mpirun lancia i processi distribuiti, OMP_NUM_THREADS controlla i thread locali
    out=$(OMP_NUM_THREADS="$OMP_THREADS_PER_RANK" \
          mpirun -np "$MPI_RANKS" $MPIRUN_EXTRA_ARGS "$binary" "$@")

    declare -g "${label}_TIME=$(echo "$out" | extract_time)"
    declare -g "${label}_CHK=$(echo "$out" | extract_checksum)"
    declare -g "${label}_RAY=$(echo "$out" | extract_rayleigh)"

    local t_var="${label}_TIME" c_var="${label}_CHK" r_var="${label}_RAY"
    echo "  -> Tempo:     ${!t_var} s"
    echo "  -> Checksum:  ${!c_var}"
    echo "  -> Rayleigh:  ${!r_var}"
}

compare_to_seq() {
    local label="$1"
    local dump_file="$2"

    local seq_chk_var="SEQ_CHK" seq_ray_var="SEQ_RAY"
    local par_chk_var="${label}_CHK" par_ray_var="${label}_RAY"

    echo "--- $label vs SEQ ---"

    if [ "${!seq_chk_var}" == "${!par_chk_var}" ]; then
        echo "  [INFO] Checksum: Bitwise identici (${!seq_chk_var})"
    else
        echo "  [INFO] Checksum: Differenti (normale per via dell'ordine dei float in parallelo)"
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

    if [ "$ENABLE_DUMP" = true ]; then
        if [ -f "$SEQ_DUMP_FILE" ] && [ -f "$dump_file" ]; then
            if cmp -s "$SEQ_DUMP_FILE" "$dump_file"; then
                echo "  [OK] Dump Vector: File bitwise identici!"
            else
                local max_err vec_check
                max_err=$(paste "$SEQ_DUMP_FILE" "$dump_file" | awk -v tol="$TOLERANCE" '
                    BEGIN { max_diff = 0.0 }
                    { d = $1 - $2; if (d < 0) d = -d; if (d > max_diff) max_diff = d }
                    END { printf "%.17e", max_diff }
                ')
                vec_check=$(awk -v e="$max_err" -v tol="$TOLERANCE" \
                    'BEGIN { print (e <= tol) ? "PASS" : "FAIL" }')

                if [ "$vec_check" == "PASS" ]; then
                    echo "  [OK] Dump Vector (L_inf norm): VALIDATO (Max diff: $max_err <= $TOLERANCE)"
                else
                    echo "  [ERRORE] Dump Vector (L_inf norm): FUORI TOLLERANZA! (Max diff: $max_err > $TOLERANCE)"
                fi
            fi
        else
            echo "  [ERRORE] File di dump non trovati!"
        fi
    fi
    echo ""
}

# ==========================================
# 4. Riepilogo configurazione
# ==========================================
echo "=========================================================="
echo " CONFIGURAZIONE TEST"
echo "=========================================================="
echo "  Matrice (N x N):        $N"
echo "  Elementi non-nulli (NZ): $NZ"
echo "  Modalità matrice:       $MODE"
echo "  Seed:                   $SEED"
echo "  Thread (single-node):   $THREADS"
echo "  SpMV Chunk Size (-c):   $CHUNK_SIZE"
echo "  Norm Chunk Size (-nc):  $NORM_CHUNK"
echo "  MPI nodes:              $MPI_NODES"
echo "  MPI ranks:              $MPI_RANKS ($RANKS_PER_NODE ranks/node)"
echo "  OMP threads/rank:       $OMP_THREADS_PER_RANK"
echo "  Tolleranza numerica:    $TOLERANCE"
echo "=========================================================="

# ==========================================
# 5. Esecuzione Sequenziale
# ==========================================
SEQ_DUMP_FLAG=""
[ "$ENABLE_DUMP" = true ] && SEQ_DUMP_FLAG="--dump-vector $SEQ_DUMP_FILE"

echo "Esecuzione Sequenziale..."
run_and_extract "SEQ" "$SEQ_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" $SEQ_DUMP_FLAG
echo "----------------------------------------------------------"

# ==========================================
# 6. Esecuzione versioni a singolo nodo
# ==========================================
for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file <<< "$entry"
    dump_flag=""
    [ "$ENABLE_DUMP" = true ] && dump_flag="--dump-vector $dump_file"

    echo "Esecuzione $label (-t $THREADS, -c $CHUNK_SIZE, -nc $NORM_CHUNK)..."
    run_and_extract "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -t "$THREADS" -c "$CHUNK_SIZE" -nc "$NORM_CHUNK" -s "$SEED" $dump_flag
    echo "----------------------------------------------------------"
done

# ==========================================
# 7. Esecuzione versioni MPI+OpenMP
# ==========================================
for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file <<< "$entry"
    dump_flag=""
    [ "$ENABLE_DUMP" = true ] && dump_flag="--dump-vector $dump_file"

    echo "Esecuzione $label (mpirun -np $MPI_RANKS, OMP_NUM_THREADS=$OMP_THREADS_PER_RANK)..."
    run_and_extract_mpi "$label" "$binary" -n "$N" -nz "$NZ" -m "$MODE" \
        -c "$CHUNK_SIZE" -nc "$NORM_CHUNK" -s "$SEED" $dump_flag
    echo "----------------------------------------------------------"
done

# ==========================================
# 8. Verifica di Correttezza e Tempi
# ==========================================
echo "=========================================================="
echo " RISULTATI CORRETTEZZA (Tolleranza: $TOLERANCE)"
echo "=========================================================="

for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file <<< "$entry"
    compare_to_seq "$label" "$dump_file"
done

for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file <<< "$entry"
    compare_to_seq "$label" "$dump_file"
done

echo "=========================================================="
echo " RIEPILOGO TEMPI"
echo "=========================================================="
printf "  %-15s %10s s\n" "SEQ" "$SEQ_TIME"
for entry in "${IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file <<< "$entry"
    t_var="${label}_TIME"
    printf "  %-15s %10s s\n" "$label" "${!t_var}"
done
for entry in "${MPI_IMPLS[@]}"; do
    IFS='|' read -r label binary dump_file <<< "$entry"
    t_var="${label}_TIME"
    printf "  %-15s %10s s\n" "$label" "${!t_var}"
done
echo "=========================================================="
