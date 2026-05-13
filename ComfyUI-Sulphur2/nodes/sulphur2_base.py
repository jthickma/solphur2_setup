from __future__ import annotations

import logging

from .constants import GEMMA3_TEXT_ENCODER, SPATIAL_UPSCALER, SULPHUR_CKPT
from .utils import invoke_node, merge_model_list

log = logging.getLogger("ComfyUI-Sulphur2")


class Sulphur2BaseLoader:
    CATEGORY = "Sulphur2/loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "AUDIO_VAE", "LATENT_UPSCALE_MODEL")
    RETURN_NAMES = ("model", "clip", "vae", "audio_vae", "upscale_model")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (merge_model_list("checkpoints", [SULPHUR_CKPT]), {"default": SULPHUR_CKPT}),
                "text_encoder": (
                    merge_model_list("text_encoders", [GEMMA3_TEXT_ENCODER]),
                    {"default": GEMMA3_TEXT_ENCODER},
                ),
                "upscale_model_name": (
                    merge_model_list("upscale_models", [SPATIAL_UPSCALER]),
                    {"default": SPATIAL_UPSCALER},
                ),
                "text_encoder_device": (["default", "cpu", "cuda"], {"default": "default"}),
                "sage_attention": (
                    ["sageattn_qk_int8_pv_fp16_cuda", "auto", "disabled"],
                    {"default": "sageattn_qk_int8_pv_fp16_cuda"},
                ),
                "allow_compile": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        ckpt_name: str,
        text_encoder: str,
        upscale_model_name: str,
        text_encoder_device: str,
        sage_attention: str,
        allow_compile: bool,
    ):
        model, _unused_clip, vae = invoke_node("CheckpointLoaderSimple", ckpt_name=ckpt_name)[:3]
        if sage_attention != "disabled":
            try:
                model = invoke_node(
                    "PathchSageAttentionKJ",
                    model=model,
                    sage_attention=sage_attention,
                    allow_compile=allow_compile,
                )[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("SageAttention patch failed, continuing with unpatched model: %s", exc)

        clip = invoke_node(
            "LTXAVTextEncoderLoader",
            text_encoder=text_encoder,
            ckpt_name=ckpt_name,
            device=text_encoder_device,
        )[0]
        audio_vae = invoke_node("LTXVAudioVAELoader", ckpt_name=ckpt_name)[0]
        upscale_model = invoke_node("LatentUpscaleModelLoader", model_name=upscale_model_name)[0]
        return (model, clip, vae, audio_vae, upscale_model)


class Sulphur2TextEncoder:
    CATEGORY = "Sulphur2/loaders"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (merge_model_list("checkpoints", [SULPHUR_CKPT]), {"default": SULPHUR_CKPT}),
                "text_encoder": (
                    merge_model_list("text_encoders", [GEMMA3_TEXT_ENCODER]),
                    {"default": GEMMA3_TEXT_ENCODER},
                ),
                "device": (["default", "cpu", "cuda"], {"default": "default"}),
            }
        }

    def load(self, ckpt_name: str, text_encoder: str, device: str):
        return invoke_node(
            "LTXAVTextEncoderLoader",
            text_encoder=text_encoder,
            ckpt_name=ckpt_name,
            device=device,
        )
