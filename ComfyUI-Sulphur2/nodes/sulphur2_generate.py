from __future__ import annotations

import secrets

from .constants import DEFAULT_NEGATIVE_PROMPT, MAX_SEED
from .sulphur2_base import Sulphur2BaseLoader
from .sulphur2_conditioning import Sulphur2Conditioning
from .sulphur2_decode import Sulphur2DecodeAndSave
from .sulphur2_prompt_enhancer import Sulphur2PromptEnhancer
from .sulphur2_sampler import Sulphur2Stage1Sampler, Sulphur2Stage2Sampler
from .sulphur2_spatial import Sulphur2SpatialUpscale
from .utils import snap_dim_32, snap_frames


class Sulphur2Generate:
    CATEGORY = "Sulphur2"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE_PROMPT}),
                "width": ("INT", {"default": 1280, "min": 512, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 704, "min": 320, "max": 1088, "step": 32}),
                "duration_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.25}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "mode": (["quality", "fast"], {"default": "quality"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "randomize_seed": ("BOOLEAN", {"default": True}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "video/solphur2"}),
            }
        }

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        duration_seconds: float,
        fps: float,
        mode: str,
        seed: int,
        randomize_seed: bool,
        enhance_prompt: bool,
        filename_prefix: str,
    ):
        seed = secrets.randbelow(MAX_SEED) if randomize_seed or int(seed) == 0 else int(seed)
        width = snap_dim_32(width)
        height = snap_dim_32(height)
        frames = snap_frames(round(float(duration_seconds) * float(fps)))

        prompt = Sulphur2PromptEnhancer().enhance(
            prompt=prompt,
            enhancer_url="http://enhancer:8080/v1/chat/completions",
            enabled=enhance_prompt,
            timeout_seconds=180,
        )[0]

        model, clip, vae, audio_vae, upscale_model = Sulphur2BaseLoader().load(
            ckpt_name="sulphur_dev_fp8mixed.safetensors",
            text_encoder="gemma_3_12B_it_fp8_scaled.safetensors",
            upscale_model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            text_encoder_device="default",
            sage_attention="sageattn_qk_int8_pv_fp16_cuda",
            allow_compile=False,
        )
        positive, negative = Sulphur2Conditioning().encode(
            clip=clip,
            positive=prompt,
            negative=negative_prompt,
            fps=fps,
        )
        stage1 = Sulphur2Stage1Sampler().sample(
            model=model,
            audio_vae=audio_vae,
            positive=positive,
            negative=negative,
            seed=seed,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            mode=mode,
            stage1_lora_name="ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
        )[0]
        video_latent, audio_latent, _combined = Sulphur2SpatialUpscale().upscale(
            av_latent=stage1,
            upscale_model=upscale_model,
            vae=vae,
        )
        stage2 = Sulphur2Stage2Sampler().sample(
            model=model,
            video_latent=video_latent,
            audio_latent=audio_latent,
            positive=positive,
            negative=negative,
            seed=seed,
            mode=mode,
            stage2_lora_name="ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
        )[0]
        return Sulphur2DecodeAndSave().decode(
            av_latent=stage2,
            vae=vae,
            audio_vae=audio_vae,
            fps=fps,
            filename_prefix=filename_prefix,
            format="mp4",
            codec="h264",
            horizontal_tiles=2,
            vertical_tiles=2,
            overlap=6,
            last_frame_fix=False,
        )
