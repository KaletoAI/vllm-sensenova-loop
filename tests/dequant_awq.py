"""Dequantize SenseNova's AWQ release into a BF16 checkpoint (for reference runs in transformers).

Usage: dequant_awq.py <awq-dir> <bf16-source-dir> <out-dir>
Linear layers and the LM head use AutoAWQ's GEMM layout: qweight [in, out/8] int32 with the
nibble order 0,2,4,6,1,3,5,7, qzeros [in/group, out/8] in the same order, scales [in/group, out];
W[out, in] = ((q - z) * s)^T. The token embedding is row-major (vllm#64832): qweight [vocab, hidden/8]
with nibbles in order 0..7, qzeros [vocab, groups/8], scales [vocab, groups].
Every other tensor is taken from the AWQ checkpoint; config, tokenizer, code and mtp.safetensors
come from the BF16 source. Prints the relative error of each dequantized tensor against the source
(a wrong nibble order shows up as an error near 1.4 instead of ~0.1).
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

awq, src, out = sys.argv[1:4]
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)


def unpack(packed, order):
    nib = (packed.unsqueeze(-1) >> SHIFTS) & 15  # [..., n, 8], nibble k at bits 4k
    if order is not None:
        inv = [order.index(i) for i in range(8)]  # column j of the 8 holds nibble AWQ_ORDER-inverse
        nib = nib[..., inv]
    return nib.reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def dequant_gemm(qw, qz, sc):
    q = unpack(qw, AWQ_ORDER).float()  # [in, out]
    z = unpack(qz, AWQ_ORDER).float()  # [in/g, out]
    g = q.shape[0] // z.shape[0]
    w = (q - z.repeat_interleave(g, 0)) * sc.float().repeat_interleave(g, 0)
    return w.t().contiguous()


def dequant_rowmajor(qw, qz, sc):
    q = unpack(qw, None).float()  # [vocab, hidden]
    z = unpack(qz, None).float()  # [vocab, groups]
    g = q.shape[1] // z.shape[1]
    return (q - z.repeat_interleave(g, 1)) * sc.float().repeat_interleave(g, 1)


def open_all(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    files = {f: safe_open(os.path.join(d, f), "pt") for f in set(idx.values())}
    return idx, files


aidx, afiles = open_all(awq)
sidx, sfiles = open_all(src)
aget = lambda k: afiles[aidx[k]].get_tensor(k)
sget = lambda k: sfiles[sidx[k]].get_tensor(k)

os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        p = os.path.join(src, f)
        (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(out, f), **({"dirs_exist_ok": True} if os.path.isdir(p) else {}))
shutil.copy2(os.path.join(src, "mtp.safetensors"), os.path.join(out, "mtp.safetensors"))

tensors, errs, other_diff = {}, [], []
for k in sorted(aidx):
    if k.endswith((".qzeros", ".scales")):
        continue
    if k.endswith(".qweight"):
        base = k[: -len(".qweight")]
        qw, qz, sc = aget(k), aget(base + ".qzeros"), aget(base + ".scales")
        w = (dequant_rowmajor if base.endswith("embed_tokens") else dequant_gemm)(qw, qz, sc)
        name = base + ".weight"
        ref = sget(name).float()
        assert w.shape == ref.shape, (name, w.shape, ref.shape)
        errs.append(((w - ref).norm() / ref.norm()).item())
        tensors[name] = w.to(torch.bfloat16)
    else:
        t = aget(k)
        if k in sidx:
            r = sget(k)
            if t.dtype != r.dtype or not torch.equal(t.to(r.dtype), r):
                other_diff.append(k)
        tensors[k] = t
missing = sorted(set(sidx) - set(tensors))
print(f"{len(errs)} AWQ tensors dequantized: rel. error vs source median {sorted(errs)[len(errs)//2]:.4f}, "
      f"max {max(errs):.4f}; {len(tensors) - len(errs)} other tensors, {len(other_diff)} differ from the source "
      f"{other_diff[:5]}; missing vs source: {len(missing)} {missing[:5]}")

shard, size, n, wmap = {}, 0, 0, {}
def flush():
    global shard, size, n
    if shard:
        n += 1
        fn = f"model-{n:05d}.safetensors"
        save_file(shard, os.path.join(out, fn), metadata={"format": "pt"})
        wmap.update({k: fn for k in shard})
        shard, size = {}, 0
for k, t in tensors.items():
    shard[k] = t.contiguous()
    size += t.numel() * t.element_size()
    if size > 4e9:
        flush()
flush()
json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
print(f"wrote {n} shards to {out}")
