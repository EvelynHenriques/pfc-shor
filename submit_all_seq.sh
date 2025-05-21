#!/bin/bash
set -euo pipefail

NS=(39 63 119 217 221 341 362 417 597 681 721 959 993 1101 1138 1247 1345 1502 1689 1821 1894 1959 1961)
MAX_N=2048
BIG_N_LIMIT=1000
TOTAL=1
INTERNAL=1
MAX_SEQ=5
HRT="48:00:00"
LOGDIR="$HOME/shor/versao2/logs_seq"
mkdir -p "$LOGDIR"

# Conta apenas os jobs seq_ ativos
count_seq() {
  qstat -u "$USER" | awk '$3 ~ /^seq_/ {c++} END{print c+0}'
}

for N in "${NS[@]}"; do
  (( N > MAX_N )) && continue

  JOB="seq_N${N}"
  LOG="$LOGDIR/${JOB}.txt"

  if (( N > BIG_N_LIMIT )); then
    # Grandes rodam sozinhos na fila seq_: espera 0 seq_
    while (( $(count_seq) != 0 )); do
      echo "[BIG SEQ] Aguardando fila seq_ livre ($(count_seq) ativos)"
      sleep 120
    done
    echo "[BIG SEQ] Submetendo $JOB (exclusivo na seq_)"
    qsub -N "$JOB" -pe smp "$TOTAL" -cwd \
         -l h_rt=$HRT -l exclusive \
         -o "$LOG" \
         ./shor.qsub "$N" "$TOTAL" "$INTERNAL"

  else
    # Pequenos limitados a MAX_SEQ na fila seq_
    while (( $(count_seq) >= MAX_SEQ )); do
      echo "[SEQ] Aguardando… ($(count_seq)/$MAX_SEQ) seq_ ativos"
      sleep 60
    done
    echo "[SEQ] Submetendo $JOB"
    qsub -N "$JOB" -pe smp "$TOTAL" -cwd \
         -l h_rt=$HRT \
         -o "$LOG" \
         ./shor.qsub "$N" "$TOTAL" "$INTERNAL"
  fi

done
