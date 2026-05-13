from __future__ import annotations

from .constants import DEFAULT_NEGATIVE_PROMPT
from .utils import invoke_node


class Sulphur2Conditioning:
    CATEGORY = "Sulphur2/conditioning"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE_PROMPT}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
            }
        }

    def encode(self, clip, positive: str, negative: str, fps: float):
        positive_encoded = invoke_node("CLIPTextEncode", text=positive, clip=clip)[0]
        negative_encoded = invoke_node("CLIPTextEncode", text=negative, clip=clip)[0]
        return invoke_node(
            "LTXVConditioning",
            positive=positive_encoded,
            negative=negative_encoded,
            frame_rate=float(fps),
        )[:2]
