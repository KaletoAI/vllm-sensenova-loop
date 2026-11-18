# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree vLLM plugin for SenseNova Loop (sensenova/SenseNova-Loop-9B).

Registered through the ``vllm.general_plugins`` entry point; vLLM calls
``register()`` in every process before configs are parsed.
"""

ARCH = "SenseNovaLoopForConditionalGeneration"
MODEL_TYPE = "sensenova_loop"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_sensenova_loop.config import SenseNovaLoopConfig
    from vllm_sensenova_loop.verify import SenseNovaLoopVerifyConfig

    # vLLM resolves model_type -> config class through _CONFIG_REGISTRY before
    # falling back to transformers' AutoConfig (no trust_remote_code needed).
    if MODEL_TYPE not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY[MODEL_TYPE] = SenseNovaLoopConfig
    MODELS_CONFIG_MAP.setdefault(ARCH, SenseNovaLoopVerifyConfig)
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCH, "vllm_sensenova_loop.model:SenseNovaLoopForConditionalGeneration"
        )
