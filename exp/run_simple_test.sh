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
# 5. Verifica di Correttezza
# ==========================================
echo "RISULTATI CORRETTEZZA:"

# Controllo 1: Checksum
if [ "$SEQ_CHK" == "$THR_CHK" ]; then
    echo "  [OK] Checksum: IDENTICI ($SEQ_CHK)"
else
    echo "  [MISMATCH] Checksum differenti! (Seq: $SEQ_CHK, Thr: $THR_CHK)"
fi

# Controllo 2: Valore di Rayleigh
DIFF_RAY=$(awk -v s="$SEQ_RAY" -v t="$THR_RAY" 'BEGIN { diff = s - t; if (diff < 0) diff = -diff; printf "%.1e", diff }')
echo "  [INFO] Diff Rayleigh: $DIFF_RAY"

# Controllo 3: Confronto file Dump Vettori (se abilitato)
if [ "$ENABLE_DUMP" = true ]; then
    if [ -f "$SEQ_DUMP_FILE" ] && [ -f "$THR_DUMP_FILE" ]; then
        if cmp -s "$SEQ_DUMP_FILE" "$THR_DUMP_FILE"; then
            echo "  [OK] Dump Vector: File BITWISE IDENTICI!"
        else
            # Calcola l'errore massimo componente per componente (L_inf norm)
            MAX_ERR=$(paste "$SEQ_DUMP_FILE" "$THR_DUMP_FILE" | awk '
                BEGIN { max_diff = 0.0 }
                {
                    diff = $1 - $2;
                    if (diff < 0) diff = -diff;
                    if (diff > max_diff) max_diff = diff;
                }
                END { printf "%.17e", max_diff }
            ')
            echo "  [WARN] Dump Vector non bitwise identici."
            echo "         Max Absolute Difference (L_inf): $MAX_ERR"
        fi
    else
        echo "  [ERRORE] File di dump non trovati!"
    fi
fi
echo "=========================================================="
