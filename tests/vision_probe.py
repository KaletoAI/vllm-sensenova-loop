"""Image probes against a running server with the plugin; saves exact token ids for tests/vision_ref_hf.py.

Usage: vision_probe.py <port> <served-model-name> <cases.json> <out.json>
cases.json: [{"id": ..., "images": ["/abs/a.png", ...], "question": ..., "text_first": false}, ...]
            ("image": "/abs/a.png" is accepted for a single image)
Greedy, thinking off, 64 new tokens, logprobs of the chosen tokens. Afterwards all cases are sent
again concurrently: the answers must equal the sequential ones (state separation under batching).
"""
import base64
import concurrent.futures as cf
import json
import mimetypes
import sys
import urllib.request

port, model, cases_path, out_path = sys.argv[1:5]
cases = json.load(open(cases_path))


def images_of(c):
    return c.get("images") or [c["image"]]


def ask(c):
    parts = []
    for path in images_of(c):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        url = f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
        parts.append({"type": "image_url", "image_url": {"url": url}})
    text = {"type": "text", "text": c["question"]}
    content = [text] + parts if c.get("text_first") else parts + [text]
    body = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "max_tokens": 64, "logprobs": True, "top_logprobs": 2,
        "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


results = []
for c in cases:
    r = ask(c)
    ch = r["choices"][0]
    results.append({**c, "images": images_of(c), "answer": ch["message"]["content"],
                    "prompt_token_ids": r["prompt_token_ids"], "token_ids": ch["token_ids"],
                    "logprobs": [t["logprob"] for t in ch["logprobs"]["content"]],
                    "top2_margin": [t["top_logprobs"][0]["logprob"] - t["top_logprobs"][1]["logprob"]
                                    if len(t["top_logprobs"]) > 1 else None for t in ch["logprobs"]["content"]],
                    "usage": r["usage"]})
    print(f"### {c['id']}: {c['question']}\n{ch['message']['content']}\n", flush=True)

with cf.ThreadPoolExecutor(len(cases)) as ex:
    parallel = list(ex.map(ask, cases))
same = [p["choices"][0]["token_ids"] == r["token_ids"] for p, r in zip(parallel, results)]
print(f"parallel == sequential: {sum(same)}/{len(same)}", flush=True)
# vLLM is not batch-invariant: a near-tie may flip under batching. A large sequential
# top-1/top-2 margin at the first differing token would point to leaking state instead.
for p, r, ok in zip(parallel, results, same):
    if not ok:
        a_, b_ = r["token_ids"], p["choices"][0]["token_ids"]
        i = next((k for k in range(min(len(a_), len(b_))) if a_[k] != b_[k]), min(len(a_), len(b_)))
        m = r["top2_margin"][i] if i < len(r["top2_margin"]) else None
        r["parallel_divergence"] = {"position": i, "sequential_top2_margin": m}
        print(f"  {r['id']}: first difference at token {i}, sequential top-1/top-2 margin {m}", flush=True)
json.dump({"results": results, "parallel_identical": same}, open(out_path, "w"), indent=1, ensure_ascii=False)
