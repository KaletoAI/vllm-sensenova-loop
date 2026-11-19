# SPDX-License-Identifier: Apache-2.0
"""MTP speculative decoding for SenseNova Loop.

The checkpoint's MTP head (``mtp.*``: fc, two pre-fc norms, one full-attention layer,
norm) is a plain Qwen3.5 MTP layer, so vLLM's ``Qwen3_5MTP`` runs it unchanged. It is
fed the target's final, normed hidden state: SenseNova's own MTP draft for the AWQ
build carries ``mtp.main_norm`` = the main model's final norm for the same purpose.
Embeddings and the LM head are shared with the target by vLLM's proposer.
"""

import os
from collections.abc import Iterable

import torch
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

# vLLM only configures handlers for the "vllm" logger hierarchy.
logger = init_logger("vllm.plugins.sensenova_loop.mtp")
_KV_SCALES = (".attn.q_scale", ".attn.k_scale", ".attn.v_scale", ".attn.prob_scale")

MTP_ARCH = "SenseNovaLoopMTP"
MTP_FILE = "mtp.safetensors"
MTP_DIR = "mtp"


class SenseNovaLoopMTP(Qwen3_5MTP):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        # The AWQ release ships its MTP head separately under mtp/ and declares
        # mtp_num_hidden_layers = 0 in the main config; every release has one layer.
        text_config = vllm_config.model_config.hf_text_config
        if not getattr(text_config, "mtp_num_hidden_layers", 0):
            text_config.mtp_num_hidden_layers = 1
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # A quantized (AWQ) token embedding cannot be loaded into the draft's plain
        # embedding; vLLM replaces the draft's embedding with the target's anyway.
        weights = (
            (name, w) for name, w in weights
            if not (".embed_tokens." in name and not name.endswith(".weight"))
        )
        loaded = super().load_weights(weights)
        if "model.fc.weight" not in loaded:
            # SenseNova's BF16 release keeps the head in mtp.safetensors without
            # listing it in model.safetensors.index.json, and vLLM only reads the
            # files the index names. Load that file directly.
            loaded |= super().load_weights(
                (name, w) for name, w in self._mtp_file_weights()
                # SenseNova's separate AWQ draft applies the target's final norm
                # itself; vLLM already hands the draft the normed hidden state.
                if name != "mtp.main_norm.weight"
            )
        # embed_tokens / lm_head are replaced by the target's; KV-cache scales are
        # quant-method sentinels that only exist in KV-quantized checkpoints.
        missing = sorted(
            name for name, _ in self.named_parameters()
            if name not in loaded and name.startswith("model.")
            and not name.startswith("model.embed_tokens")
            and not name.endswith(_KV_SCALES)
        )
        if missing:
            raise ValueError(
                f"SenseNova Loop MTP: {len(missing)} parameters were not loaded, "
                f"e.g. {missing[:5]}"
            )
        return loaded

    def _mtp_file_weights(self) -> Iterable[tuple[str, torch.Tensor]]:
        from safetensors import safe_open

        spec = self.vllm_config.speculative_config
        model_config = (spec and spec.draft_model_config) or self.vllm_config.model_config
        source = model_config.model_weights or model_config.model
        paths = self._mtp_files(source, model_config.revision)
        logger.info("SenseNova Loop: loading the MTP head from %s", paths)
        for path in paths:
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def _mtp_files(self, source: str, revision: str | None) -> list[str]:
        """mtp.safetensors (BF16 release) or the safetensors under mtp/ (AWQ release)."""
        if os.path.isdir(source):
            if os.path.isfile(os.path.join(source, MTP_FILE)):
                return [os.path.join(source, MTP_FILE)]
            sub = os.path.join(source, MTP_DIR)
            files = sorted(
                os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".safetensors")
            ) if os.path.isdir(sub) else []
            if not files:
                raise FileNotFoundError(
                    f"SenseNova Loop MTP: no mtp.* weights in the index and neither "
                    f"{MTP_FILE} nor {MTP_DIR}/*.safetensors in {source}"
                )
            return files
        from huggingface_hub import HfApi, hf_hub_download

        names = HfApi().list_repo_files(source, revision=revision)
        wanted = [MTP_FILE] if MTP_FILE in names else sorted(
            n for n in names if n.startswith(MTP_DIR + "/") and n.endswith(".safetensors")
        )
        if not wanted:
            raise FileNotFoundError(f"SenseNova Loop MTP: no MTP weights in {source}")
        return [
            hf_hub_download(
                source, n, revision=revision,
                cache_dir=self.vllm_config.load_config.download_dir,
            )
            for n in wanted
        ]


# Saved at import, before patch_speculative_config() replaces it.
_original_hf_config_override = SpeculativeConfig.hf_config_override


def hf_config_override(hf_config):
    """SpeculativeConfig.hf_config_override with SenseNova Loop -> SenseNovaLoopMTP.

    Module-level so the draft ModelConfig that stores it stays picklable.
    """
    if getattr(hf_config, "model_type", None) == "sensenova_loop":
        text_config = getattr(hf_config, "text_config", hf_config)
        # The AWQ release keeps its MTP head under mtp/ and declares 0 layers in the
        # main config; one layer is what every release ships. A checkpoint without
        # any MTP weights fails clearly when SenseNovaLoopMTP loads them.
        n_predict = getattr(text_config, "mtp_num_hidden_layers", None) or 1
        hf_config.model_type = "qwen3_5_mtp"
        hf_config.update({"n_predict": n_predict, "architectures": [MTP_ARCH]})
        return hf_config
    return _original_hf_config_override(hf_config)


def patch_speculative_config() -> None:
    if SpeculativeConfig.hf_config_override is not hf_config_override:
        SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)
