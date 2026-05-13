from __future__ import annotations

from .constants import (
    DISTILL_LORA,
    DISTILL_LORA_STRENGTH_STAGE1,
    DISTILL_LORA_STRENGTH_STAGE2,
)
from .utils import invoke_node, merge_model_list


class Sulphur2DistillLoRALoader:
    CATEGORY = "Sulphur2/loaders"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (merge_model_list("loras", [DISTILL_LORA]), {"default": DISTILL_LORA}),
                "stage": (["stage_1_fast", "stage_2"], {"default": "stage_2"}),
                "strength_override": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05},
                ),
            }
        }

    def load(self, model, lora_name: str, stage: str, strength_override: float):
        if strength_override >= 0:
            strength = float(strength_override)
        elif stage == "stage_1_fast":
            strength = DISTILL_LORA_STRENGTH_STAGE1
        else:
            strength = DISTILL_LORA_STRENGTH_STAGE2
        return invoke_node(
            "LoraLoaderModelOnly",
            model=model,
            lora_name=lora_name,
            strength_model=strength,
        )[:1]
