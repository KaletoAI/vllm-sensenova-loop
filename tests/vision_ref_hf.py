"""Check vLLM image answers against SenseNova's own transformers code (needs transformers 5.18).

Usage: vision_ref_hf.py <hf-model-dir> <probe.json from vision_probe.py> [out.json] [--min-top1 0.9]
Per probe:
  1. HF's processor must build exactly vLLM's prompt token ids (image tokens, template, order).
  2. HF runs vLLM's exact sequence (prompt + greedy answer) with the same images, no cache, and
     reports top-1 agreement with vLLM's tokens, mean/max |logprob gap| of those tokens, and the
     HF margin (top-1 minus vLLM's token) wherever they disagree: a near-tie gives a small margin.
Exits non-zero if a prompt differs or top-1 agreement falls below --min-top1.
"""
import argparse
import json
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("probe")
ap.add_argument("out", nargs="?")
ap.add_argument("--min-top1", type=float, default=0.9)
a = ap.parse_args()
data = json.load(open(a.probe))
probes = data["results"] if isinstance(data, dict) else data
proc = AutoProcessor.from_pretrained(a.src, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    a.src, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()

rows, failures = [], []
for p in probes:
    paths = p.get("images") or [p["image"]]
    content = [{"type": "image"} for _ in paths]
    text_part = {"type": "text", "text": p["question"]}
    content = [text_part] + content if p.get("text_first") else content + [text_part]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=[text], images=[Image.open(x).convert("RGB") for x in paths],
                  return_tensors="pt").to("cuda:0")
    hf_prompt = inputs["input_ids"][0].tolist()
    same_prompt = hf_prompt == p["prompt_token_ids"]
    comp = p["token_ids"]
    ids = torch.tensor([p["prompt_token_ids"] + comp], device="cuda:0")
    # M-RoPE needs the token types for the whole sequence; the prompt part must match the processor's.
    mm_types = (ids == model.config.image_token_id).long()
    types_ok = "mm_token_type_ids" not in inputs or torch.equal(
        mm_types[0, :len(hf_prompt)], inputs["mm_token_type_ids"][0].long()) if same_prompt else False
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inputs["pixel_values"],
                       image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=mm_types,
                       use_cache=False).logits[0].float()
    n = len(p["prompt_token_ids"])
    lp = torch.log_softmax(logits[n - 1:n - 1 + len(comp)], -1)
    tgt = torch.tensor(comp, device="cuda:0")
    chosen = lp.gather(1, tgt[:, None])[:, 0]
    agree_mask = lp.argmax(-1) == tgt
    agree = agree_mask.float().mean().item()
    gaps = (chosen.cpu() - torch.tensor(p["logprobs"])).abs()
    margins = (lp.max(-1).values - chosen)[~agree_mask].tolist()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    hf_answer = proc.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    row = dict(id=p["id"], same_prompt=same_prompt, types_ok=types_ok, prompt_len=n, top1_agree=agree,
               mean_abs_logprob_gap=gaps.mean().item(), max_abs_logprob_gap=gaps.max().item(),
               disagree_margins=[round(m, 3) for m in margins], vllm=p["answer"], hf=hf_answer)
    rows.append(row)
    ok = same_prompt and types_ok and agree >= a.min_top1
    if not ok:
        failures.append(p["id"])
    print(f"### {p['id']} [{'OK' if ok else 'FAIL'}]: prompt ids identical={same_prompt} ({n}), "
          f"token types ok={types_ok}, top-1 {agree:.1%}, |Δlogprob| mean {gaps.mean():.3f} "
          f"max {gaps.max():.3f}, margins at disagreements {row['disagree_margins']}\n"
          f"  vLLM: {p['answer'][:160]!r}\n  HF:   {hf_answer[:160]!r}\n", flush=True)
if isinstance(data, dict) and "parallel_identical" in data:
    print(f"parallel == sequential (from probe): {sum(data['parallel_identical'])}/{len(data['parallel_identical'])}")
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1, ensure_ascii=False)
print("FAILED: " + ", ".join(failures) if failures else "ALL OK")
sys.exit(1 if failures else 0)
