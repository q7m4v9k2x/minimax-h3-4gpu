#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-reports/gpu-samples.csv}"
mkdir -p "$(dirname "$OUT")"
echo 'timestamp,index,name,util_gpu,util_mem,mem_used_mib,mem_total_mib,clock_sm_mhz,power_w,throttle' > "$OUT"
echo "Sampling GPUs; press Ctrl-C after the render completes: $OUT"
while true; do
  ts="$(date -Is)"
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits \
    | sed "s/^/$ts,/" >> "$OUT"
  sleep "${INTERVAL:-1}"
done

