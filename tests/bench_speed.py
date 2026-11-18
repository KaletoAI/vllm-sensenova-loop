"""Quick throughput check against a running OpenAI-compatible server.

Prefill: ~4.8K-token prompt, max_tokens=1 (median of 3). Decode: 256 tokens, 1 and 4 parallel streams.
Usage: bench_speed.py <tag> <served-model-name> [port]   (default port 9150)
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

TAG, MODEL = sys.argv[1], sys.argv[2]
URL = f"http://127.0.0.1:{sys.argv[3] if len(sys.argv) > 3 else 9150}/v1/completions"
long_prompt = "The history of the city is long and full of change. " * 400


def req(prompt, max_tokens, ignore_eos=True):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": ignore_eos}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(r, timeout=600))
    return time.time() - t0, out["usage"]


req("warmup", 8)
req(long_prompt[:2000], 4)

res = {}
pf = []
for i in range(3):
    dt, u = req(f"[{i}] " + long_prompt, 1)
    pf.append(u["prompt_tokens"] / dt)
res["prefill_tok_s"] = round(sorted(pf)[1], 1)

dt, u = req("Write a long story about a lighthouse keeper.", 256)
res["decode_1x_tok_s"] = round(u["completion_tokens"] / dt, 1)

with cf.ThreadPoolExecutor(4) as ex:
    t0 = time.time()
    outs = list(ex.map(lambda i: req(f"Story {i}: a lighthouse keeper", 256), range(4)))
    wall = time.time() - t0
res["decode_4x_total_tok_s"] = round(sum(u["completion_tokens"] for _, u in outs) / wall, 1)
print(TAG, json.dumps(res))
