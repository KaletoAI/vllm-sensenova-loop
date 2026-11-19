# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova Loop (fixed loop count, optional mHC) with the Qwen3.5 vision tower."""

from collections.abc import Iterable

import re
from collections import defaultdict

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.model_executor.models.interfaces import HasInnerState
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from vllm_sensenova_loop.awq import AWQ_QUANTIZATION, awq_embedding_method, extend_awq_skips, is_awq
from vllm_sensenova_loop.config import SenseNovaLoopConfig
from vllm_sensenova_loop.mhc import MHCRouter


class SenseNovaLoopDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str) -> None:
        super().__init__(vllm_config, layer_type, prefix)
        config = vllm_config.model_config.hf_text_config
        self.physical_index = config.loop_physical_indices[self.layer_idx]
        self.mhc = config.mhc
        self.mhc_active = self.mhc is not None and (
            self.mhc["start"] <= self.physical_index < self.mhc["end"]
        )
        if self.mhc_active:
            self.mhc_attn = MHCRouter(config.hidden_size, self.mhc)
            self.mhc_ffn = MHCRouter(config.hidden_size, self.mhc)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.mhc_active:
            return super().forward(hidden_states, residual, positions, **kwargs)
        if residual is not None:
            hidden_states = hidden_states + residual
        streams = self.mhc["streams"]
        if self.physical_index == self.mhc["start"]:
            hidden_states = hidden_states.unsqueeze(-2).expand(-1, streams, -1)
        else:
            hidden_states = hidden_states.unflatten(-1, (streams, -1))
        branch, post, mix = self.mhc_attn(hidden_states)
        branch = self.input_layernorm(branch.contiguous())
        if self.layer_type == "linear_attention":
            branch = self.linear_attn(hidden_states=branch)
        else:
            branch = self.self_attn(hidden_states=branch, positions=positions)
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        branch, post, mix = self.mhc_ffn(hidden_states)
        branch = self.mlp(self.post_attention_layernorm(branch.contiguous()))
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        if self.physical_index == self.mhc["end"] - 1:
            hidden_states = hidden_states.mean(-2)
        else:
            hidden_states = hidden_states.flatten(-2)
        return hidden_states, None


class SenseNovaLoopModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.config.hidden_size,
            quant_method=awq_embedding_method(self.quant_config, self.config),
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            SenseNovaLoopDecoderLayer(vllm_config, layer_type, f"{prefix}.layers.{i}")
            for i, layer_type in enumerate(self.config.layer_types)
        )
        self.norm = Qwen3_5RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions)
        if residual is None:
            return self.norm(hidden_states)
        return self.norm(hidden_states, residual)[0]

    def share_loop_parameters(self) -> None:
        first: dict[int, nn.Module] = {}
        for physical, layer in zip(self.config.loop_physical_indices, self.layers):
            if physical not in first:
                first[physical] = layer
                continue
            for name, param in first[physical].named_parameters():
                parent, _, leaf = name.rpartition(".")
                setattr(layer.get_submodule(parent), leaf, param)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=Qwen3_5Model.hf_to_vllm_mapper
        )


SUPPORTED_QUANTIZATION = {"modelopt_fp4", AWQ_QUANTIZATION}

# Matches decoder-layer entries in both naming schemes: the checkpoint's
# ``model.language_model.layers.N`` after the multimodal weights mapper
# (``language_model.model.layers.N``) and the text-only ``model.layers.N``.
_LAYER_ENTRY = re.compile(r"((?:language_model\.)?model\.layers)\.(\d+)(\..*)?")


def remap_exclude_modules(quant_config, loop_indices) -> None:
    """Translate a ModelOpt exclude list from checkpoint to vLLM module names.

    The checkpoint names physical layers (``model.language_model.layers.30``);
    vLLM builds one module per execution (``language_model.model.layers.0..55``)
    so every loop visit owns its cache. Each physical-layer entry is expanded to
    all of its executions. ``prefix*`` subtree wildcards are split into
    ``prefix`` and ``prefix.*`` first, as ModelOptQuantConfigBase.apply_vllm_mapper
    does; vLLM has usually applied the model's weights mapper already.
    """
    exclude = getattr(quant_config, "exclude_modules", None)
    if not exclude or getattr(quant_config, "_sensenova_loop_remapped", False):
        return
    executions: dict[int, list[int]] = defaultdict(list)
    for execution, physical in enumerate(loop_indices):
        executions[physical].append(execution)
    remapped: set[str] = set()
    for pattern in exclude:
        if len(pattern) >= 2 and pattern[-1] == "*" and pattern[-2] != ".":
            patterns = [pattern[:-1], pattern[:-1] + ".*"]
        else:
            patterns = [pattern]
        for p in patterns:
            # Entries the weights mapper left untouched (no trailing dot to match).
            if p.startswith("model.language_model."):
                p = "language_model.model." + p[len("model.language_model."):]
            elif p in ("model.visual", "visual"):
                p = "visual"
            elif p == "lm_head":
                # Keep the bare name too: the MTP draft builds its own "lm_head"
                # with this quant config (it is then replaced by the target's).
                remapped.add(p)
                p = "language_model.lm_head"
            m = _LAYER_ENTRY.fullmatch(p)
            if m is None:
                remapped.add(p)
                continue
            for execution in executions.get(int(m.group(2)), ()):
                # A whole-layer entry becomes "layers.<e>.*": a bare "layers.3" would
                # also hit layers 30-39 through ModelOpt's substring fallback.
                remapped.add(f"{m.group(1)}.{execution}{m.group(3) or '.*'}")
    quant_config.exclude_modules = sorted(remapped)
    quant_config._sensenova_loop_remapped = True


class SenseNovaLoopForCausalLM(nn.Module):
    """The looped text decoder with its LM head (``language_model`` of the VLM)."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_text_config
        self.model = SenseNovaLoopModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        quant_config = vllm_config.quant_config
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            # The AWQ release ships an INT4 LM head; ModelOpt builds keep it in BF16.
            quant_config=quant_config if is_awq(quant_config) else None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Expects ``model.*`` / ``lm_head.*`` names (prefix already stripped)."""
        destinations: dict[int, list[int]] = {}
        for execution, physical in enumerate(self.config.loop_physical_indices):
            destinations.setdefault(physical, []).append(execution)

        def expanded_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name == "model.loop_gate.logits":  # unused by the fixed-count loop
                    continue
                if name.startswith("model.layers."):
                    _, _, index, suffix = name.split(".", 3)
                    for execution in destinations[int(index)]:
                        yield f"model.layers.{execution}.{suffix}", weight
                else:
                    yield name, weight

        return AutoWeightsLoader(self).load_weights(expanded_weights())


class SenseNovaLoopProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(SenseNovaLoopConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=SenseNovaLoopProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class SenseNovaLoopForConditionalGeneration(Qwen3_5ForConditionalGeneration, HasInnerState):
    """SenseNova Loop with its Qwen3.5 vision tower.

    Everything multimodal (vision tower, HF processor, M-RoPE positions, merging
    image embeddings into the token embeddings) is Qwen3.5's, as in SenseNova's
    own modeling code; only ``language_model`` is the looped decoder. SenseNova
    uses no deepstack (``deepstack_visual_indexes`` is empty).
    """

    # Qwen3.5's mapper: model.visual. -> visual., model.language_model. ->
    # language_model.model., lm_head. -> language_model.lm_head., mtp. dropped.
    # vLLM also applies it to a ModelOpt exclude list before __init__ runs.
    hf_to_vllm_mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
    # Fused projections, passed by reference to the quant config so that exclude
    # lists written per checkpoint shard match the fused modules.
    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        speculative = vllm_config.speculative_config
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or (speculative is not None and speculative.method != "mtp")
            or vllm_config.lora_config is not None
            or getattr(text_config, "layer_scale", False)
        ):
            raise ValueError(
                "SenseNova Loop currently requires inference without pipeline "
                "parallelism, LoRA or layer_scale; speculation only via MTP"
            )
        quant_config = vllm_config.quant_config
        if quant_config is not None:
            if quant_config.get_name() not in SUPPORTED_QUANTIZATION:
                raise ValueError(
                    f"SenseNova Loop supports {sorted(SUPPORTED_QUANTIZATION)} "
                    f"checkpoints, got {quant_config.get_name()!r}"
                )
            remap_exclude_modules(quant_config, text_config.loop_physical_indices)
            extend_awq_skips(quant_config)
        multimodal_config = vllm_config.model_config.get_multimodal_config()

        # vLLM's MTP proposer reads image_token_index from configs it does not
        # recognise by class name; Qwen3.5 configs call it image_token_id.
        if getattr(config, "image_token_index", None) is None:
            config.image_token_index = config.image_token_id
        self.config = config
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self.video_pruning_method = None  # read by Qwen3-VL's EVS path (video only)
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # Attributes read by the inherited Qwen3-VL embedding/EVS helpers.
        self.use_deepstack = bool(
            getattr(config.vision_config, "deepstack_visual_indexes", None)
        )
        if self.use_deepstack:
            raise ValueError("SenseNova Loop checkpoints with deepstack are not supported")
        self.deepstack_num_level = 0
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = 0

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = SenseNovaLoopForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        # vLLM only checks for unloaded parameters on unquantized checkpoints. Several
        # of ours (mHC routers, vision tower) start as torch.empty, so a naming slip
        # in a quantized checkpoint would otherwise serve garbage without an error.
        # Attention KV-cache scales (attn.{q,k,v,prob}_scale) are created by the quant
        # method with a -1 sentinel and are absent unless the KV cache is quantized.
        kv_scales = (".attn.q_scale", ".attn.k_scale", ".attn.v_scale", ".attn.prob_scale")
        missing = sorted(
            name for name, _ in self.named_parameters()
            if name not in loaded and not name.endswith(kv_scales)
        )
        if missing:
            raise ValueError(
                f"SenseNova Loop: {len(missing)} parameters were not loaded from the "
                f"checkpoint, e.g. {missing[:5]}"
            )
        return loaded

    def make_empty_intermediate_tensors(self, *args, **kwargs):
        raise NotImplementedError("SenseNova Loop does not support pipeline parallelism")

    def process_weights_after_loading(self) -> None:
        # Parameters share storage; attention modules and their caches stay distinct.
        self.language_model.model.share_loop_parameters()
