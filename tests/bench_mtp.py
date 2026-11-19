"""Decode speed on realistic chat tasks (for MTP on/off comparisons).

Usage: bench_mtp.py <tag> <served-model-name> [port] [out.jsonl]
Six chat prompts (EN/DE, code, explanation, story, list), thinking off, up to 384 new tokens.
Two sampling settings: greedy, and SenseNova's temperature 1.0 / top_p 0.95 plus top_k 20
(min_p is not available with speculative decoding in vLLM 0.31; the script records the error).
For each: single stream (prompts one after another) and 4 concurrent streams (total tok/s).
Prompts are short, so time is dominated by decoding. With MTP, the mean acceptance length is
read from the server's Prometheus counters (1 + accepted / drafts).
"""
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.request

TAG, MODEL = sys.argv[1], sys.argv[2]
PORT = sys.argv[3] if len(sys.argv) > 3 else "9150"
OUT = sys.argv[4] if len(sys.argv) > 4 else None
BASE = f"http://127.0.0.1:{PORT}"
PROMPTS = [
    "Explain how a refrigerator works, step by step, for a curious 12-year-old.",
    "Erkläre ausführlich, wie die Photosynthese funktioniert und warum sie für das Leben auf der Erde wichtig ist.",
    "Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short usage example.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Schreibe eine Anleitung, wie man einen Sauerteig ansetzt und ein erstes Brot damit backt.",
    "List the main differences between TCP and UDP and give two typical use cases for each.",
]
SETTINGS = {
    "greedy": {"temperature": 0.0},
    "sampled": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
}


def chat(prompt, sampling):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 384,
            "chat_template_kwargs": {"enable_thinking": False}, "seed": 1, **sampling}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return time.time() - t0, r["usage"]["completion_tokens"]


def spec_counters():
    try:
        text = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    except Exception:
        return None
    vals = {}
    for key in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total"):
        m = [float(x) for x in re.findall(rf"^{re.escape(key)}(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M)]
        vals[key] = sum(m) if m else None
    return vals


def acceptance(before, after):
    if not before or not after or None in before.values() or None in after.values():
        return None
    drafts = after["vllm:spec_decode_num_drafts_total"] - before["vllm:spec_decode_num_drafts_total"]
    acc = after["vllm:spec_decode_num_accepted_tokens_total"] - before["vllm:spec_decode_num_accepted_tokens_total"]
    return round(1 + acc / drafts, 3) if drafts else None


# Warmup: a real prompt (prefill > 32 tokens) and one 4-stream round, so JIT and
# CUDA-graph capture for larger batches do not land in the first measurement.
chat(PROMPTS[0] + " Use at least three paragraphs.", SETTINGS["greedy"])
with cf.ThreadPoolExecutor(4) as ex:
    list(ex.map(lambda p: chat(p, SETTINGS["greedy"]), PROMPTS[2:6]))
res = {"tag": TAG}
for name, sampling in SETTINGS.items():
    c0 = spec_counters()
    total_t = total_n = 0
    for p in PROMPTS:
        t, n = chat(p, sampling)
        total_t += t
        total_n += n
    c1 = spec_counters()
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda p: chat(p, sampling), PROMPTS[:4]))
    wall = time.time() - t0
    c2 = spec_counters()
    res[name] = {"tok_s_1x": round(total_n / total_t, 1), "tok_s_4x_total": round(sum(n for _, n in outs) / wall, 1),
                 "tokens_1x": total_n, "accept_len_1x": acceptance(c0, c1), "accept_len_4x": acceptance(c1, c2)}
try:
    chat("Hi", {"temperature": 1.0, "min_p": 0.05})
    res["min_p_request"] = "ok"
except urllib.error.HTTPError as e:
    res["min_p_request"] = f"HTTP {e.code}: {e.read().decode()[:200]}"
print(json.dumps(res), flush=True)
if OUT:
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")
