# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree vLLM plugin for SenseNova Loop (sensenova/SenseNova-Loop-9B).

Registered through the ``vllm.general_plugins`` entry point; vLLM calls
``register()`` in every process before configs are parsed.
"""

ARCH = "SenseNovaLoopForConditionalGeneration"
MODEL_TYPE = "sensenova_loop"
AWQ_ARCH_ALIAS = "Qwen3_5LoopForConditionalGeneration"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_sensenova_loop.config import SenseNovaLoopConfig
    from vllm_sensenova_loop.mtp import MTP_ARCH, patch_speculative_config
    from vllm_sensenova_loop.verify import SenseNovaLoopVerifyConfig

    # vLLM resolves model_type -> config class through _CONFIG_REGISTRY before
    # falling back to transformers' AutoConfig (no trust_remote_code needed).
    if MODEL_TYPE not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY[MODEL_TYPE] = SenseNovaLoopConfig
    # SenseNova's AWQ release names the architecture Qwen3_5LoopForConditionalGeneration.
    for arch in (ARCH, AWQ_ARCH_ALIAS):
        MODELS_CONFIG_MAP.setdefault(arch, SenseNovaLoopVerifyConfig)
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                arch, "vllm_sensenova_loop.model:SenseNovaLoopForConditionalGeneration"
            )
    # MTP speculative decoding: the draft config resolves to SenseNovaLoopMTP.
    if MTP_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(MTP_ARCH, "vllm_sensenova_loop.mtp:SenseNovaLoopMTP")
    patch_speculative_config()
