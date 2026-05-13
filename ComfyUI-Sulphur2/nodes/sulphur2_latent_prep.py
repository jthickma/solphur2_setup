from __future__ import annotations

from .utils import snap_dim_32, snap_frames, stage1_dim


class Sulphur2LatentPrep:
    CATEGORY = "Sulphur2/utils"
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "stage1_width", "stage1_height", "frames", "fps")
    FUNCTION = "prepare"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1280, "min": 512, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 704, "min": 320, "max": 1088, "step": 32}),
                "duration_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.25}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
            }
        }

    def prepare(self, width: int, height: int, duration_seconds: float, fps: float):
        final_width = snap_dim_32(width)
        final_height = snap_dim_32(height)
        frames = snap_frames(round(float(duration_seconds) * float(fps)))
        return (
            final_width,
            final_height,
            stage1_dim(final_width),
            stage1_dim(final_height),
            frames,
            float(fps),
        )
