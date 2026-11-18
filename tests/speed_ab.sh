#!/bin/bash
# Serve each variant with the plugin (vllm 0.31.0), run greedy probes + bench_speed.py, stop.
# Usage: speed_ab.sh name=model_dir [name=model_dir ...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=${BENCH:-$HERE/bench_speed.py}
for spec in "$@"; do
  name=${spec%%=*}; dir=${spec#*=}
  echo "=== $name ($dir) $(date -Is)"
  KEEP=1 PORT=9150 "$HERE/serve_smoke.sh" "$dir" "$name" 2>&1 | grep -E "READY|FAILED|Model loading|KV cache|NVFP4 GEMM|###|^ [0-9]+ *$"
  if curl -sf localhost:9150/health >/dev/null; then
    python3 "$BENCH" "$name" snl
  fi
  docker stop "snl-$name" >/dev/null 2>&1
  sleep 5
done
echo "=== SPEED-AB FERTIG $(date -Is)"
