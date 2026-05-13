from __future__ import annotations

from .constants import VAE_HORIZONTAL_TILES, VAE_TILE_OVERLAP, VAE_VERTICAL_TILES
from .utils import invoke_node


class Sulphur2DecodeAndSave:
    CATEGORY = "Sulphur2/output"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "decode"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "vae": ("VAE",),
                "audio_vae": ("AUDIO_VAE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "filename_prefix": ("STRING", {"default": "video/solphur2"}),
                "format": (["mp4", "mov"], {"default": "mp4"}),
                "codec": (["h264", "hevc", "prores"], {"default": "h264"}),
                "horizontal_tiles": ("INT", {"default": VAE_HORIZONTAL_TILES, "min": 1, "max": 8}),
                "vertical_tiles": ("INT", {"default": VAE_VERTICAL_TILES, "min": 1, "max": 8}),
                "overlap": ("INT", {"default": VAE_TILE_OVERLAP, "min": 1, "max": 8}),
                "last_frame_fix": ("BOOLEAN", {"default": False}),
            }
        }

    def decode(
        self,
        av_latent,
        vae,
        audio_vae,
        fps: float,
        filename_prefix: str,
        format: str,
        codec: str,
        horizontal_tiles: int,
        vertical_tiles: int,
        overlap: int,
        last_frame_fix: bool,
    ):
        video_latent, audio_latent = invoke_node("LTXVSeparateAVLatent", av_latent=av_latent)[:2]
        frames = invoke_node(
            "LTXVTiledVAEDecode",
            vae=vae,
            latents=video_latent,
            horizontal_tiles=int(horizontal_tiles),
            vertical_tiles=int(vertical_tiles),
            overlap=int(overlap),
            last_frame_fix=bool(last_frame_fix),
            working_device="auto",
            working_dtype="auto",
        )[0]
        audio = invoke_node(
            "LTXVAudioVAEDecode",
            samples=audio_latent,
            audio_vae=audio_vae,
        )[0]
        video = invoke_node(
            "CreateVideo",
            images=frames,
            audio=audio,
            fps=float(fps),
        )[0]
        invoke_node(
            "SaveVideo",
            video=video,
            filename_prefix=filename_prefix,
            format=format,
            codec=codec,
        )
        return (frames, audio)
