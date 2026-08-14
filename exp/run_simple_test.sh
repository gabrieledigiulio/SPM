#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 1. Configurazione Parametri del Test
# ==========================================
SEQ_BIN="../iterative_SpMV"
CPPTHREADS_BIN="../threadpool_SpMV"

# Parametri del problema
N=500000
NZ=20000000
MODE="irregular"
SEED=111

# Parametri del modello parallelo
THREADS=16
CHUNK_SIZE=1000       # Granularità SpMV (-c)
NORM_CHUNK=0          # Granularità operazioni vettoriali (-nc, 0 = automatico)

# Tolleranza numerica (come da report del progetto: 10^-12)
TOLERANCE="1e-12"

# Controllo del dump dei vettori
ENABLE_DUMP=true      # Imposta a true per abilitare il dump e confronto vettoriale
SEQ_DUMP_FILE="seq_vec.dump"
THR_DUMP_FILE="thr_vec.dump"

# ==========================================
# 2. Funzioni di estrazione dati
# ==========================================
extract_time()      { grep -oP 'Time \(sec\) = \K[0-9.]+' | head -1; }
extract_checksum()  { grep -oP 'checksum=\K0x[0-9a-fA-F]+'; }
extract_rayleigh()  { grep -oP 'rayleigh=\K[-0-9.eE+]+'; }

SEQ_DUMP_FLAG=""
THR_DUMP_FLAG=""
if [ "$ENABLE_DUMP" = true ]; then
    SEQ_DUMP_FLAG="--dump-vector $SEQ_DUMP_FILE"
    THR_DUMP_FLAG="--dump-vector $THR_DUMP_FILE"
fi

echo "=========================================================="
echo " CONFIGURAZIONE TEST"
echo "=========================================================="
echo "  Matrice (N x N):        $N"
echo "  Elementi non-nulli (NZ): $NZ"
echo "  Modalità matrice:       $MODE"
echo "  Seed:                   $SEED"
echo "  Thread C++:             $THREADS"
echo "  SpMV Chunk Size (-c):   $CHUNK_SIZE"
echo "  Norm Chunk Size (-nc):  $NORM_CHUNK (0 = statico / automatico)"
echo "  Tolleranza numerica:    $TOLERANCE"
echo "  Verifica Dump Vettore:  $ENABLE_DUMP"
echo "=========================================================="

# ==========================================
# 3. Esecuzione Sequenziale
# ==========================================
echo -n "Esecuzione Sequenziale... "
SEQ_OUT=$($SEQ_BIN -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" $SEQ_DUMP_FLAG)
SEQ_TIME=$(echo "$SEQ_OUT" | extract_time)
SEQ_CHK=$(echo "$SEQ_OUT" | extract_checksum)
SEQ_RAY=$(echo "$SEQ_OUT" | extract_rayleigh)

echo "Completata!"
echo "  -> Tempo:     ${SEQ_TIME} s"
echo "  -> Checksum:  ${SEQ_CHK}"
echo "  -> Rayleigh:  ${SEQ_RAY}"
echo "----------------------------------------------------------"

# ==========================================
# 4. Esecuzione C++ Threads
# ==========================================
echo -n "Esecuzione C++ Threads (-t $THREADS, -c $CHUNK_SIZE, -nc $NORM_CHUNK)... "
THR_OUT=$($CPPTHREADS_BIN -n "$N" -nz "$NZ" -m "$MODE" -t "$THREADS" -c "$CHUNK_SIZE" -nc "$NORM_CHUNK" -s "$SEED" $THR_DUMP_FLAG)
THR_TIME=$(echo "$THR_OUT" | extract_time)
THR_CHK=$(echo "$THR_OUT" | extract_checksum)
THR_RAY=$(echo "$THR_OUT" | extract_rayleigh)

echo "Completata!"
echo "  -> Tempo:     ${THR_TIME} s"
echo "  -> Checksum:  ${THR_CHK}"
echo "  -> Rayleigh:  ${THR_RAY}"
echo "=========================================================="

# ==========================================
# 5. Verifica di Correttezza con Tolleranza
# ==========================================
echo "RISULTATI CORRETTEZZA (Tolleranza: $TOLERANCE):"

# Nota sul Checksum (informativa, poiché differisce a causa della non-associatività dei float)
if [ "$SEQ_CHK" == "$THR_CHK" ]; then
    echo "  [INFO] Checksum: Bitwise identici ($SEQ_CHK)"
else
    echo "  [INFO] Checksum: Differenti (normale per via dell'ordine dei float in parallelo)"
fi

# Controllo 1: Differenza sul valore di Rayleigh rispetto alla tolleranza
RAY_CHECK=$(awk -v s="$SEQ_RAY" -v t="$THR_RAY" -v tol="$TOLERANCE" 'BEGIN {
    diff = s - t;
    if (diff < 0) diff = -diff;
    if (diff <= tol) print "PASS"; else print "FAIL";
}')

DIFF_RAY_VAL=$(awk -v s="$SEQ_RAY" -v t="$THR_RAY" 'BEGIN { diff = s - t; if (diff < 0) diff = -diff; printf "%.2e", diff }')

if [ "$RAY_CHECK" == "PASS" ]; then
    echo "  [OK] Rayleigh value: VALIDATO (Diff: $DIFF_RAY_VAL <= $TOLERANCE)"
else
    echo "  [ERRORE] Rayleigh value: FUORI TOLLERANZA! (Diff: $DIFF_RAY_VAL > $TOLERANCE)"
fi

# Controllo 2: Confronto file Dump Vettori (se abilitato)
if [ "$ENABLE_DUMP" = true ]; then
    if [ -f "$SEQ_DUMP_FILE" ] && [ -f "$THR_DUMP_FILE" ]; then
        if cmp -s "$SEQ_DUMP_FILE" "$THR_DUMP_FILE"; then
            echo "  [OK] Dump Vector: File bitwise identici!"
        else
            # Calcola l'errore massimo componente per componente (L_inf norm) e confrontalo con la tolleranza
            MAX_ERR=$(paste "$SEQ_DUMP_FILE" "$THR_DUMP_FILE" | awk -v tol="$TOLERANCE" '
                BEGIN { max_diff = 0.0 }
                {
                    diff = $1 - $2;
                    if (diff < 0) diff = -diff;
                    if (diff > max_diff) max_diff = diff;
                }
                END { printf "%.17e", max_diff }
            ')
            
            VEC_CHECK=$(awk -v err="$MAX_ERR" -v tol="$TOLERANCE" 'BEGIN { if (err <= tol) print "PASS"; else print "FAIL"; }')

            if [ "$VEC_CHECK" == "PASS" ]; then
                echo "  [OK] Dump Vector (L_inf norm): VALIDATO (Max diff: $MAX_ERR <= $TOLERANCE)"
            else
                echo "  [ERRORE] Dump Vector (L_inf norm): FUORI TOLLERANZA! (Max diff: $MAX_ERR > $TOLERANCE)"
            fi
        fi
    else
        echo "  [ERRORE] File di dump non trovati!"
    fi
fi
echo "=========================================================="
