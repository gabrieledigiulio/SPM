#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 1. Configurazione Parametri (da Table 8 del Report)
# ==========================================
SEQ_BIN="../iterative_SpMV"
CPPTHREADS_BIN="../threadpool_SpMV"

# Parametri del problema identici al report
N=1000000
NZ=200000000
SEED=111

# Parametri paralleli (1 nodo, 32 thread, block size 2048 come da Table 7/8)
THREADS=32
CHUNK_SIZE=2048
NORM_CHUNK=0

# ==========================================
# 2. Funzioni di estrazione dati
# ==========================================
extract_time()      { grep -oP 'Time \(sec\) = \K[0-9.]+' | head -1; }
extract_checksum()  { grep -oP 'checksum=\K0x[0-9a-fA-F]+'; }
extract_rayleigh()  { grep -oP 'rayleigh=\K[-0-9.eE+]+'; }

echo "=========================================================="
echo " CONFRONTO REGULAR vs IRREGULAR (Table 8 del Report)"
echo "=========================================================="
echo "  Matrice (N x N):        $N"
echo "  Elementi non-nulli (NZ): $NZ"
echo "  Thread C++:             $THREADS"
echo "  SpMV Chunk Size (-c):   $CHUNK_SIZE"
echo "=========================================================="

# ==========================================
# 3. Test Modalità REGULAR
# ==========================================
echo "[1/2] Test in modalità REGULAR..."

# Sequenziale Regular
SEQ_REG_OUT=$($SEQ_BIN -n "$N" -nz "$NZ" -m "regular" -s "$SEED")
SEQ_REG_TIME=$(echo "$SEQ_REG_OUT" | extract_time)

# C++ Threads Regular
THR_REG_OUT=$($CPPTHREADS_BIN -n "$N" -nz "$NZ" -m "regular" -t "$THREADS" -c "$CHUNK_SIZE" -nc "$NORM_CHUNK" -s "$SEED")
THR_REG_TIME=$(echo "$THR_REG_OUT" | extract_time)

echo "  -> Sequenziale Regular:  ${SEQ_REG_TIME} s"
echo "  -> C++ Threads Regular:  ${THR_REG_TIME} s"
echo "----------------------------------------------------------"

# ==========================================
# 4. Test Modalità IRREGULAR
# ==========================================
echo "[2/2] Test in modalità IRREGULAR..."

# Sequenziale Irregular
SEQ_IRR_OUT=$($SEQ_BIN -n "$N" -nz "$NZ" -m "irregular" -s "$SEED")
SEQ_IRR_TIME=$(echo "$SEQ_IRR_OUT" | extract_time)

# C++ Threads Irregular
THR_IRR_OUT=$($CPPTHREADS_BIN -n "$N" -nz "$NZ" -m "irregular" -t "$THREADS" -c "$CHUNK_SIZE" -nc "$NORM_CHUNK" -s "$SEED")
THR_IRR_TIME=$(echo "$THR_IRR_OUT" | extract_time)

echo "  -> Sequenziale Irregular: ${SEQ_IRR_TIME} s"
echo "  -> C++ Threads Irregular: ${THR_IRR_TIME} s"
echo "=========================================================="

# ==========================================
# 5. Tabella Riassuntiva Finale (Stile Table 8)
# ==========================================
echo ""
echo "=========================================================="
echo " TABELLA RISULTATI (Confronto Regular vs Irregular)"
echo "=========================================================="
printf "%-18s | %-18s | %-18s\n" "Implementation" "Regular Time (sec)" "Irregular Time (sec)"
echo "----------------------------------------------------------"
printf "%-18s | %-18.2f | %-18.2f\n" "Sequential" "$SEQ_REG_TIME" "$SEQ_IRR_TIME"
printf "%-18s | %-18.2f | %-18.2f\n" "C++ threads" "$THR_REG_TIME" "$THR_IRR_TIME"
echo "=========================================================="
