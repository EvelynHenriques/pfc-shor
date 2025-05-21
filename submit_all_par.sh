#!/bin/bash
set -euo pipefail

NS=(39 63 119 217 221 341 362 417 597 681 721 959 993 1101 1138 1247 1345 1502 1689 1821 1894 1959 1961)
MAX_N=2048
BIG_N_LIMIT=1000
TOTAL_LIST=(16 32 48 64 80)
INTERNAL_LIST=(2 4 8 16)
MAX_PAR=5
HRT="02:00:00"
LOGDIR="$HOME/shor/versao2/logs_par"
mkdir -p "$LOGDIR"

# Conta apenas os jobs par_ ativos
count_par() {
  qstat -u "$USER" | awk '$3 ~ /^par_/ {c++} END{print c+0}'
}

for N in "${NS[@]}"; do
  (( N > MAX_N )) && continue
  for TOTAL in "${TOTAL_LIST[@]}"; do
    for K in "${INTERNAL_LIST[@]}"; do
      (( TOTAL % K != 0 )) && continue

      JOB="par_N${N}_T${TOTAL}_I${K}"
      LOG="$LOGDIR/${JOB}.txt"

      if (( N > BIG_N_LIMIT )); then
        # Grandes rodam sozinhos na fila par_: espera 0 par_
        while (( $(count_par) != 0 )); do
          echo "[BIG PAR] Aguardando fila par_ livre ($(count_par) ativos)"
          sleep 120
        done
        echo "[BIG PAR] Submetendo $JOB (exclusivo na par_)"
        qsub -N "$JOB" -pe smp "$TOTAL" -cwd \
             -l h_rt=$HRT -l exclusive \
             -o "$LOG" \
             ./shor.qsub "$N" "$TOTAL" "$K"

      else
        # Pequenos limitados a MAX_PAR na fila par_
        while (( $(count_par) >= MAX_PAR )); do
          echo "[PAR] Aguardando… ($(count_par)/$MAX_PAR) par_ ativos"
          sleep 60
        done
        echo "[PAR] Submetendo $JOB"
        qsub -N "$JOB" -pe smp "$TOTAL" -cwd \
             -l h_rt=$HRT \
             -o "$LOG" \
             ./shor.qsub "$N" "$TOTAL" "$K"
      fi

    done
  done
done
