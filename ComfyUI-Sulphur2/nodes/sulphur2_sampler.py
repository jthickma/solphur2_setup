from __future__ import annotations

from .constants import (
    DISTILL_LORA,
    DISTILL_LORA_STRENGTH_STAGE1,
    DISTILL_LORA_STRENGTH_STAGE2,
    MAX_SEED,
    QUALITY_CFG_STAGE1,
    QUALITY_CFG_STAGE2,
    QUALITY_LTXVSCHEDULER_BASE_SHIFT,
    QUALITY_LTXVSCHEDULER_MAX_SHIFT,
    QUALITY_LTXVSCHEDULER_STEPS,
    QUALITY_LTXVSCHEDULER_TERMINAL,
    STAGE1_SIGMAS_FAST,
    STAGE2_SIGMAS_FAST,
    STAGE2_SIGMAS_QUALITY,
)
from .utils import invoke_node, merge_model_list, stage1_dim, warn_ood_sulphur


class Sulphur2Stage1Sampler:
    CATEGORY = "Sulphur2/sampling"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "audio_vae": ("AUDIO_VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 42, "min": 0, "max": MAX_SEED}),
                "width": ("INT", {"default": 1280, "min": 512, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 704, "min": 320, "max": 1088, "step": 32}),
                "frames": ("INT", {"default": 241, "min": 9, "max": 481, "step": 8}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "mode": (["quality", "fast"], {"default": "quality"}),
                "stage1_lora_name": (merge_model_list("loras", [DISTILL_LORA]), {"default": DISTILL_LORA}),
            }
        }

    def sample(
        self,
        model,
        audio_vae,
        positive,
        negative,
        seed: int,
        width: int,
        height: int,
        frames: int,
        fps: float,
        mode: str,
        stage1_lora_name: str,
    ):
        warn_ood_sulphur(width, height, frames, fps, mode)
        stage1_width = stage1_dim(width)
        stage1_height = stage1_dim(height)

        video_latent = invoke_node(
            "EmptyLTXVLatentVideo",
            width=stage1_width,
            height=stage1_height,
            length=int(frames),
            batch_size=1,
        )[0]
        audio_latent = invoke_node(
            "LTXVEmptyLatentAudio",
            audio_vae=audio_vae,
            frames_number=int(frames),
            frame_rate=int(round(float(fps))),
            batch_size=1,
        )[0]
        av_latent = invoke_node(
            "LTXVConcatAVLatent",
            video_latent=video_latent,
            audio_latent=audio_latent,
        )[0]

        if mode == "fast":
            model = invoke_node(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=stage1_lora_name,
                strength_model=DISTILL_LORA_STRENGTH_STAGE1,
            )[0]
            sampler = invoke_node("KSamplerSelect", sampler_name="euler_ancestral_cfg_pp")[0]
            sigmas = invoke_node("ManualSigmas", sigmas=STAGE1_SIGMAS_FAST)[0]
            cfg = 1.0
        else:
            sampler = invoke_node("KSamplerSelect", sampler_name="euler_ancestral")[0]
            sigmas = invoke_node(
                "LTXVScheduler",
                steps=QUALITY_LTXVSCHEDULER_STEPS,
                max_shift=QUALITY_LTXVSCHEDULER_MAX_SHIFT,
                base_shift=QUALITY_LTXVSCHEDULER_BASE_SHIFT,
                stretch=True,
                terminal=QUALITY_LTXVSCHEDULER_TERMINAL,
                latent=av_latent,
            )[0]
            cfg = QUALITY_CFG_STAGE1

        noise = invoke_node("RandomNoise", noise_seed=int(seed))[0]
        guider = invoke_node(
            "CFGGuider",
            model=model,
            positive=positive,
            negative=negative,
            cfg=cfg,
        )[0]
        return invoke_node(
            "SamplerCustomAdvanced",
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=av_latent,
        )[:1]


class Sulphur2Stage2Sampler:
    CATEGORY = "Sulphur2/sampling"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 42, "min": 0, "max": MAX_SEED}),
                "mode": (["quality", "fast"], {"default": "quality"}),
                "stage2_lora_name": (merge_model_list("loras", [DISTILL_LORA]), {"default": DISTILL_LORA}),
            }
        }

    def sample(
        self,
        model,
        video_latent,
        audio_latent,
        positive,
        negative,
        seed: int,
        mode: str,
        stage2_lora_name: str,
    ):
        model = invoke_node(
            "LoraLoaderModelOnly",
            model=model,
            lora_name=stage2_lora_name,
            strength_model=DISTILL_LORA_STRENGTH_STAGE2,
        )[0]
        av_latent = invoke_node(
            "LTXVConcatAVLatent",
            video_latent=video_latent,
            audio_latent=audio_latent,
        )[0]
        sampler = invoke_node("KSamplerSelect", sampler_name="euler_cfg_pp")[0]
        sigmas = invoke_node(
            "ManualSigmas",
            sigmas=STAGE2_SIGMAS_FAST if mode == "fast" else STAGE2_SIGMAS_QUALITY,
        )[0]
        noise = invoke_node("RandomNoise", noise_seed=int(seed) + 1)[0]
        guider = invoke_node(
            "CFGGuider",
            model=model,
            positive=positive,
            negative=negative,
            cfg=QUALITY_CFG_STAGE2,
        )[0]
        return invoke_node(
            "SamplerCustomAdvanced",
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=av_latent,
        )[:1]
