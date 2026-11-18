# vllm-sensenova-loop

Out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin for
[SenseNova Loop](https://huggingface.co/sensenova/SenseNova-Loop-9B): a looped Qwen3.5
(hybrid Gated DeltaNet / full attention) whose middle layers run twice per forward pass,
with two-stream manifold-constrained hyper-connections (mHC).

In-tree support was proposed in [vllm-project/vllm#64832](https://github.com/vllm-project/vllm/pull/64832)
and the maintainers suggested a plugin instead. This plugin starts from that PR's
implementation (see `NOTICE`; the first commit holds the PR files unchanged) and adds:

- loading as a plugin (`vllm.general_plugins` entry point), no vLLM fork needed
- **ModelOpt NVFP4** checkpoints (`W4A16_NVFP4` and `NVFP4`): the checkpoint's
  `exclude_modules` name physical layers, vLLM builds one module per loop execution;
  the plugin expands each physical-layer entry to all of its executions
- validation against the final **SenseNova-Loop-9B** release (YaRN, 512K) — in progress

**Status: work in progress, not yet validated.** Targets `vllm==0.31.0`.

## Install

```bash
pip install --no-deps git+https://github.com/<TBD>/vllm-sensenova-loop
vllm serve sensenova/SenseNova-Loop-9B --max-model-len 32768
```

No `--trust-remote-code` needed; the plugin registers the `sensenova_loop` config and
the `SenseNovaLoopForConditionalGeneration` architecture.

## Limitations (inherited from #64832 unless noted)

- text only (vision tower and MTP head are skipped while loading)
- no speculative decoding, pipeline parallelism, LoRA, CPU offload, `sharded_state` loading
- quantization: ModelOpt FP4 only (this plugin); other methods are rejected

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
