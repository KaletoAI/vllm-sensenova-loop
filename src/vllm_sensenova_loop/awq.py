# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova's official AWQ checkpoint (sensenova/SenseNova-Loop-9B-AWQ).

The INT4 embedding method is from vllm-project/vllm#64832 (commit 202d66c,
"[Model] Support SenseNova Loop AWQ checkpoints"). The checkpoint keeps the decoder
projections, the token embedding (row-major INT4, zero point, group 128) and the
LM head in 4 bits; DeltaNet in_proj_a/b, the mHC routers and the vision tower stay
in their original precision. vLLM's AutoAWQConfig handles the linear layers.
"""

import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs

AWQ_QUANTIZATION = "auto_awq"


class SenseNovaLoopAWQEmbeddingMethod(QuantizeMethodBase):
    """Decode the row-major INT4 embedding for the requested tokens."""

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if input_size % (8 * self.group_size):
            raise ValueError("SenseNova Loop AWQ embedding dimensions are misaligned")
        num_rows = sum(output_partition_sizes)
        num_groups = input_size // self.group_size
        for name, width, dtype in (
            ("qweight", input_size // 8, torch.int32),
            ("qzeros", num_groups // 8, torch.int32),
            ("scales", num_groups, torch.float32),
        ):
            param = nn.Parameter(
                torch.empty(num_rows, width, dtype=dtype), requires_grad=False
            )
            set_weight_attrs(param, {"output_dim": 0, **extra_weight_attrs})
            layer.register_parameter(name, param)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        shifts = torch.arange(0, 32, 4, device=input_.device, dtype=torch.int32)

        def unpack(packed: torch.Tensor, width: int) -> torch.Tensor:
            words = F.embedding(input_, packed)
            return ((words.unsqueeze(-1) >> shifts) & 15).reshape(*input_.shape, width)

        groups = layer.embedding_dim // self.group_size
        weights = unpack(layer.qweight, layer.embedding_dim).unflatten(
            -1, (groups, self.group_size)
        )
        zeros = unpack(layer.qzeros, groups)
        scales = F.embedding(input_, layer.scales)
        return (
            ((weights - zeros.unsqueeze(-1)) * scales.unsqueeze(-1))
            .flatten(-2)
            .to(layer.params_dtype)
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SenseNova Loop AWQ embeddings are lookup-only")


def is_awq(quant_config) -> bool:
    return quant_config is not None and quant_config.get_name() == AWQ_QUANTIZATION


def awq_embedding_method(quant_config, text_config) -> QuantizeMethodBase | None:
    """The INT4 embedding method if the AWQ config declares a quantized embedding."""
    if not is_awq(quant_config):
        return None
    embedding = quant_config.full_config.get("embedding")
    if embedding is None:
        return None
    group_size = embedding.get("group_size")
    if (
        embedding.get("layout") != "row_major"
        or embedding.get("bits") != 4
        or embedding.get("zero_point") is not True
        or type(group_size) is not int
        or group_size <= 0
        or text_config.tie_word_embeddings
    ):
        raise ValueError(f"Unsupported SenseNova Loop AWQ embedding config: {embedding}")
    return SenseNovaLoopAWQEmbeddingMethod(group_size)


def extend_awq_skips(quant_config) -> None:
    """Keep the vision tower and the MTP layer in BF16 (they are not AWQ in the
    checkpoint). Its "model.visual" entry has no trailing dot, so vLLM's weights
    mapper leaves it alone, while vLLM names the tower "visual.*"."""
    if not is_awq(quant_config):
        return
    skips = list(quant_config.modules_to_not_convert or [])
    for name in ("visual.", "mtp."):
        if name not in skips:
            skips.append(name)
    quant_config.modules_to_not_convert = skips
