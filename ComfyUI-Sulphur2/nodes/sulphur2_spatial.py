from __future__ import annotations

from .utils import invoke_node


class Sulphur2SpatialUpscale:
    CATEGORY = "Sulphur2/sampling"
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent", "av_latent")
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            }
        }

    def upscale(self, av_latent, upscale_model, vae):
        video_latent, audio_latent = invoke_node("LTXVSeparateAVLatent", av_latent=av_latent)[:2]
        upscaled_video = invoke_node(
            "LTXVLatentUpsampler",
            samples=video_latent,
            upscale_model=upscale_model,
            vae=vae,
        )[0]
        upscaled_av = invoke_node(
            "LTXVConcatAVLatent",
            video_latent=upscaled_video,
            audio_latent=audio_latent,
        )[0]
        return (upscaled_video, audio_latent, upscaled_av)
