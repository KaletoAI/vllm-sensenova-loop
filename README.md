# vllm-sensenova-loop

Out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin for
[SenseNova Loop](https://huggingface.co/sensenova/SenseNova-Loop-9B): a looped Qwen3.5
(hybrid Gated DeltaNet / full attention) whose middle layers run twice per forward pass,
with two-stream manifold-constrained hyper-connections (mHC).

In-tree support was proposed by SenseTime in
[vllm-project/vllm#64832](https://github.com/vllm-project/vllm/pull/64832), and the maintainers
suggested a plugin instead. This plugin is built on that PR's implementation (see `NOTICE`; the
first commit holds the PR files unchanged). On top of it, it adds:

- **loading as a plugin** through the `vllm.general_plugins` entry point, with stock vLLM and no fork
- **image input.** The Qwen3.5 vision tower, processor and M-RoPE come from vLLM's own Qwen3.5
  model, exactly as SenseNova's modeling code reuses transformers' Qwen3.5; only the language model
  is the looped decoder. Checked against SenseNova's transformers code token by token (see below).
- **MTP speculative decoding** with the checkpoint's own MTP head (vLLM's Qwen3.5 MTP code), which
  gives +30 to +60 % decode speed on GB10 (see below).
- **SenseNova's official AWQ release.** The INT4 embedding method comes from the PR's AWQ commit
  (`202d66c`); the plugin adds images and MTP on top (the AWQ MTP head lives under `mtp/`).
- **ModelOpt NVFP4 checkpoints.** A checkpoint's `exclude_modules` name *physical* layers, while vLLM
  builds one module per loop *execution*, so the plugin expands every physical-layer entry to all of
  its executions. Fused projections (`qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, `in_proj_ba`) are
  mapped so that per-shard exclude lists still match.

Status: **works on vLLM 0.31.0**, tested on an NVIDIA DGX Spark (GB10) with the final
SenseNova-Loop-9B release. Early-stage community code, so bug reports are very welcome.

## Tested

| Checkpoint | Result on GB10 (vLLM 0.31.0, single GPU) |
|---|---|
| [sensenova/SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) (BF16, YaRN 512K config) | ✅ text and images, matches transformers |
| [Kaleto/SenseNova-Loop-9B-NVFP4](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4) (ModelOpt `W4A16_NVFP4`, MLP + attention + DeltaNet) | ✅ text and images, `MarlinNvFp4LinearKernel`, reasoning and tool-call parsers checked |
| [Kaleto/SenseNova-Loop-9B-NVFP4-MLP](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4-MLP) (ModelOpt `W4A16_NVFP4`, MLP only) | ✅ text and images |
| [sensenova/SenseNova-Loop-9B-AWQ](https://huggingface.co/sensenova/SenseNova-Loop-9B-AWQ) @ `64bd16ea` (official AWQ, INT4 embedding and LM head) | ✅ text, images and MTP, AWQ Marlin kernel |
| an NVFP4 checkpoint that also quantizes `in_proj_a/b` (fused `in_proj_ba`) | ✅ loads and runs |

Speed on one GB10 with `tests/bench_mtp.py`: six chat tasks (English and German, code, explanation,
story, list), thinking off, up to 384 new tokens, greedy, BF16 KV cache. Decode tok/s for one stream
and in total for four concurrent streams:

| Checkpoint | Weights in memory | No MTP, 1 / 4 streams | MTP k=1, 1 / 4 streams | MTP acceptance (k=1) |
|---|---:|---:|---:|---:|
| BF16 | 16.9 GiB | 7.5 / 22 | 12.2 / 31 | 70 % |
| NVFP4 (full decoder) | 7.7 GiB | 21.3 / 75 | 27.9 / 99 | 68 % |
| NVFP4-MLP | 10.5 GiB | 12.3 / 50 | 20.1 / 67 | 70 % |
| SenseNova AWQ | 5.5 GiB | 27.5 / 77 | 39.4 / 104 | 69 % |

k=2 and k=3 accept more tokens per step but are not faster on GB10 (BF16 k=2 is the exception:
13.0 tok/s). With temperature 1.0 / top_p 0.95 / top_k 20 single-stream speed drops by ~4 %.
MTP does not change the output: greedy answers with and without MTP are token-identical on all image
probes, and text answers differ only at exact or near ties (top-1/top-2 logprob margin ≤ 0.125).

The table below is the older single-prompt benchmark (`tests/bench_speed.py`: ~4.8K-token prompt,
256 decoded tokens, no MTP).

| Checkpoint | Weights in memory | Prefill tok/s | Decode, 1 stream | Decode, 4 streams (total) |
|---|---:|---:|---:|---:|
| BF16 | 16.9 GiB | 1,785 | 7.5 tok/s | 31.1 tok/s |
| NVFP4 (full decoder) | 7.7 GiB | 1,492 | 21.2 tok/s | 81.1 tok/s |
| NVFP4-MLP | 10.5 GiB | 1,563 | 12.4 tok/s | 54.6 tok/s |

Images, compared with SenseNova's own transformers code (BF16) on 7 probes: shapes, OCR, a bar
chart with a German question, a photo, a 1280×640 banner, two images in one message, and text
before the image. For each probe vLLM's prompt token ids are identical to the HF processor's, and HF
is run teacher-forced on vLLM's greedy answer:

| Checkpoint in vLLM | Same top token as HF BF16 | \|Δ logprob\| of vLLM's tokens: mean per probe / max |
|---|---:|---:|
| BF16 | 94–100 % (every miss is an exact tie) | 0.000–0.017 / 0.12 |
| NVFP4 | 93–100 % | 0.04–0.09 / 1.42 |
| NVFP4-MLP | 88–100 % | 0.02–0.16 / 3.13* |

\* At that token the NVFP4-MLP build reads "SENSENOVA" correctly where BF16 reads "SENENA".

Answers to all seven probes are correct for all three checkpoints. Sending the probes concurrently
gives the same tokens as sending them one by one, except at near-ties (top-1/top-2 margin ≤ 0.125),
which batching can flip in vLLM. The vision tower stays BF16 in the NVFP4 checkpoints (+0.85 GiB).
Text speed is unchanged by the vision support.

Not tested yet: video input, MTP with images under load beyond the probes above, tensor parallelism (the upstream code supports it), the Preview checkpoint,
`NVFP4` W4A4 checkpoints, and GPUs other than GB10.

## Install

Into an environment that already has `vllm==0.31.0`:

```bash
pip install --no-deps https://github.com/KaletoAI/vllm-sensenova-loop/archive/refs/heads/master.tar.gz
vllm serve sensenova/SenseNova-Loop-9B --max-model-len 32768
```

No `--trust-remote-code` is needed: the plugin registers the `sensenova_loop` config and the
`SenseNovaLoopForConditionalGeneration` architecture. On CUDA it defaults to eager compilation
with CUDA graphs for decode only. Explicit `--compilation-config` or `--enforce-eager` take precedence.

With the official image, install the plugin at container start:

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash vllm/vllm-openai:v0.31.0 -c "
    pip install -q --no-deps https://github.com/KaletoAI/vllm-sensenova-loop/archive/refs/heads/master.tar.gz &&
    exec vllm serve Kaleto/SenseNova-Loop-9B-NVFP4 \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --max-model-len 32768 --max-num-seqs 3"
```

Thinking is on by default. Turn it off per request with
`"chat_template_kwargs": {"enable_thinking": false}`.

MTP speculative decoding (recommended: one speculative token):

```bash
vllm serve Kaleto/SenseNova-Loop-9B-NVFP4 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

With speculative decoding, vLLM 0.31 rejects requests that set `min_p` or `logit_bias` (HTTP 400).
Use `top_k` instead, e.g. `top_k: 20` with SenseNova's temperature 1.0 / top_p 0.95.

## Limitations

Inherited from #64832 unless noted:

- MTP: only `method: mtp` speculation; `min_p` / `logit_bias` are unavailable while it is on (vLLM)
- video input goes through the same Qwen3.5 code but is untested; `--limit-mm-per-prompt '{"video": 0}'` turns it off
- no pipeline parallelism, LoRA, CPU offload, or `sharded_state` loading
- quantization: ModelOpt NVFP4 and SenseNova's AWQ layout (added by this plugin). Other methods are rejected.

## Tests

`tests/serve_smoke.sh <model_dir> <name>` serves a checkpoint with the plugin in the vLLM 0.31.0
image and runs four greedy chat prompts. `tests/vision_probe.py` sends image prompts to a running
server and `tests/vision_ref_hf.py` checks the answers against SenseNova's transformers code
(transformers 5.18, exit code 1 on a mismatch). `tests/bench_mtp.py` measures decode speed and MTP
acceptance on chat tasks. `tests/dequant_awq.py` turns the AWQ release into a BF16 checkpoint for
reference runs in transformers. `tests/speed_ab.sh name=dir ...` does the same and then runs
`tests/bench_speed.py`.

## Credits

The model implementation comes from [vllm-project/vllm#64832](https://github.com/vllm-project/vllm/pull/64832)
by SenseTime ([sufubao](https://github.com/sufubao)), including the AWQ embedding method. The model is by
[SenseNova](https://huggingface.co/sensenova). This plugin only packages that implementation and
extends it.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
