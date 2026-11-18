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
- **ModelOpt NVFP4 checkpoints.** A checkpoint's `exclude_modules` name *physical* layers, while vLLM
  builds one module per loop *execution*, so the plugin expands every physical-layer entry to all of
  its executions. Fused projections (`qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, `in_proj_ba`) are
  mapped so that per-shard exclude lists still match.

Status: **works on vLLM 0.31.0**, tested on an NVIDIA DGX Spark (GB10) with the final
SenseNova-Loop-9B release. Early-stage community code, so bug reports are very welcome.

## Tested

| Checkpoint | Result on GB10 (vLLM 0.31.0, single GPU) |
|---|---|
| [sensenova/SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) (BF16, YaRN 512K config) | ✅ serves, correct greedy answers |
| [Kaleto/SenseNova-Loop-9B-NVFP4](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4) (ModelOpt `W4A16_NVFP4`, MLP + attention + DeltaNet) | ✅ `MarlinNvFp4LinearKernel`, reasoning and tool-call parsers checked |
| [Kaleto/SenseNova-Loop-9B-NVFP4-MLP](https://huggingface.co/Kaleto/SenseNova-Loop-9B-NVFP4-MLP) (ModelOpt `W4A16_NVFP4`, MLP only) | ✅ |
| an NVFP4 checkpoint that also quantizes `in_proj_a/b` (fused `in_proj_ba`) | ✅ loads and runs |

Speed on one GB10: ~4.8K-token prompt, 256 decoded tokens, greedy, BF16 KV cache, no speculative
decoding. Measured with `tests/bench_speed.py`.

| Checkpoint | Weights in memory | Prefill tok/s | Decode, 1 stream | Decode, 4 streams (total) |
|---|---:|---:|---:|---:|
| BF16 | 16.9 GiB | 1,785 | 7.5 tok/s | 31.1 tok/s |
| NVFP4 (full decoder) | 7.7 GiB | 1,492 | 21.2 tok/s | 81.1 tok/s |
| NVFP4-MLP | 10.5 GiB | 1,563 | 12.4 tok/s | 54.6 tok/s |

Not tested yet: tensor parallelism (the upstream code supports it), the Preview checkpoint,
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

## Limitations

Inherited from #64832 unless noted:

- text only: the vision tower and the MTP head are skipped while loading
- no speculative decoding (MTP), pipeline parallelism, LoRA, CPU offload, or `sharded_state` loading
- quantization: ModelOpt FP4 only (added by this plugin). Other methods are rejected.

MTP and vision are the obvious next steps. Help is welcome.

## Tests

`tests/serve_smoke.sh <model_dir> <name>` serves a checkpoint with the plugin in the vLLM 0.31.0
image and runs four greedy chat prompts. `tests/speed_ab.sh name=dir ...` does the same and then runs
`tests/bench_speed.py`.

## Credits

The model implementation comes from [vllm-project/vllm#64832](https://github.com/vllm-project/vllm/pull/64832)
by SenseTime ([sufubao](https://github.com/sufubao)). The model is by
[SenseNova](https://huggingface.co/sensenova). This plugin only packages that implementation and
extends it.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
