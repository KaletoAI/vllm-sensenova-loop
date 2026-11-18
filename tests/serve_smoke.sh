#!/bin/bash
# Serve a SenseNova Loop checkpoint with the plugin in the official vLLM 0.31.0 image and run
# a few greedy chat prompts.  Usage: serve_smoke.sh <model_dir> <name> [extra vllm args...]
# Plugin source is expected at $PLUGIN (default: dir above this script).
set -u
MODEL=$(realpath "$1"); NAME=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=${PLUGIN:-$(dirname "$HERE")}
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971}
PORT=${PORT:-9160}
CACHE=${CACHE:-$HOME/.cache/vllm031-normal}
OUT=${OUT:-$HERE/out}; mkdir -p "$OUT"
LOG=$OUT/serve-$NAME.log
docker rm -f "snl-$NAME" >/dev/null 2>&1
docker run -d --rm --init --name "snl-$NAME" --network host --ipc private --shm-size 8g --gpus all \
  --ulimit memlock=-1:-1 \
  -v "$MODEL:/model:ro" -v "$PLUGIN:/plugin-src:ro" -v "$CACHE:/cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/cache/huggingface \
  -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm -e FLASHINFER_WORKSPACE_BASE=/cache \
  -e TRITON_CACHE_DIR=/cache/triton -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -c "
    cp -r /plugin-src /tmp/plugin && rm -rf /tmp/plugin/upstream-pr /tmp/plugin/tests/out &&
    pip install -q --no-deps --no-build-isolation /tmp/plugin &&
    python3 -m vllm.entrypoints.openai.api_server --model /model --served-model-name snl \
      --max-model-len ${MAXLEN:-8192} --gpu-memory-utilization ${GPUMEM:-0.3} --max-num-seqs 4 \
      --port $PORT --host 127.0.0.1 $*" >/dev/null
docker logs -f "snl-$NAME" > "$LOG" 2>&1 &
ok=0
for i in $(seq 1 120); do
  curl -sf localhost:$PORT/health >/dev/null && { ok=1; echo "READY after $((i * 10))s"; break; }
  docker ps -q -f name="snl-$NAME" | grep -q . || break
  sleep 10
done
grep -h -o -E "Loading plugin[^\n]*|plugin[s]? [^ ]*sensenova[^\n]*|Using [A-Za-z0-9]+ for NVFP4 GEMM|Selected [A-Za-z0-9]+ (kernel|backend)[^\n]*|Model loading took.*|Available KV cache memory.*|GPU KV cache size.*" "$LOG" | sort -u
if [ $ok = 1 ]; then
  python3 - "$PORT" "$OUT/answers-$NAME.json" <<'EOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
prompts = [
    "Warum ist der Himmel blau? Antworte in drei Sätzen.",
    "What is 17 * 23? Answer with the number only.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Name the capital of Australia and one fact about it.",
]
res = []
for p in prompts:
    body = {"model": "snl", "messages": [{"role": "user", "content": p}], "temperature": 0,
            "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    txt = r["choices"][0]["message"]["content"]
    res.append({"prompt": p, "answer": txt, "usage": r["usage"]})
    print("###", p, "\n", txt[:600], "\n")
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
EOF
else
  echo "SERVE FAILED"; tail -60 "$LOG"
fi
[ "${KEEP:-0}" = 1 ] || docker stop "snl-$NAME" >/dev/null 2>&1
echo "=== DONE $NAME $(date -Is)"
