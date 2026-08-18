#!/usr/bin/env bash
# ==============================================================================
# SPM Project - Configurazione Comune & Funzioni Helper
# ==============================================================================

# Percorsi Binari
export SEQ_BIN="../iterative_SpMV"
export CPP_BIN="../threadpool_SpMV"
export OMP_BIN="../omp_tasks_SpMV"
export MPI_BIN="../mpi_omp_SpMV"

# Parametri Problema (Taglia Large)
export DEFAULT_N=1000000
export DEFAULT_NZ=40000000
export DEFAULT_MODE="irregular"
export DEFAULT_SEED=111

# Risorse di default
export DEFAULT_THREADS=16
export DEFAULT_MPI_NODES=8
export DEFAULT_MPI_RANKS=8
export REPEATS=3
export BLOCK_SIZES=(256 512 1024 2048 4096 8192 16384)

# Directory Risultati
export RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
export CSV_FILE="$RESULTS_DIR/granularity_unified_results.csv"

# Funzioni di estrazione
extract_comp_time()   { grep -oP '(?:Computation time|Time) \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_spmv_time()   { grep -oP 'SpMV time \(sec\) = \K[0-9.]+' | head -1 || true; }
extract_vecops_time() { grep -oP 'Vector ops time \(sec\) = \K[0-9.]+' | head -1 || true; }

# Funzione Calcolo Mediana
calculate_median() {
    local arr=("$@")
    printf '%s\n' "${arr[@]}" | sort -n | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) print 0;
            else if (NR % 2 == 1) print a[(NR + 1) / 2];
            else print (a[NR / 2] + a[NR / 2 + 1]) / 2;
        }
    '
}