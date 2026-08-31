#!/usr/bin/env bash
set -euo pipefail

SEP=$(printf '=%.0s' {1..58})

SEQ_BIN="../iterative_SpMV"
SRUN_TIME="00:30:00"

N=1000000
NZ=250000000
MODE="irregular"
SEED=111
REPEATS=3

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_OUTPUT="${RESULT_DIR}/sequential_results.csv"

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

echo "$SEP"
echo " SEQUENTIAL BASELINE ($REPEATS REPEATS)"
echo "$SEP"
echo "  Matrix (N x NZ):      $N x $NZ"
echo "  Mode:                 $MODE"
echo "  Seed:                 $SEED"
echo "$SEP"

echo "Implementation,Total_Time_Med" > "$CSV_OUTPUT"

tot_times=()

for r in $(seq 1 "$REPEATS"); do
    echo "  [Run $r/$REPEATS]"

    out_seq=$(srun --time="$SRUN_TIME" -N 1 -n 1 -c 1 \
        "$SEQ_BIN" -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED")

    run_time=$(echo "$out_seq" | extract_tot_time)
    tot_times+=("$run_time")

    echo "  >> Tot=${run_time}s"
done

m_tot=$(calculate_median "${tot_times[@]}")

echo "SEQUENTIAL,$m_tot" >> "$CSV_OUTPUT"

echo "$SEP"
echo " Completed! Median Tot=${m_tot}s"
echo " Results saved to: $CSV_OUTPUT"
echo "$SEP"