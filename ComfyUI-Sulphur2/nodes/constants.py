SULPHUR_CKPT = "sulphur_dev_fp8mixed.safetensors"
DISTILL_LORA = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
SPATIAL_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA3_TEXT_ENCODER = "gemma_3_12B_it_fp8_scaled.safetensors"

DISTILL_LORA_STRENGTH_STAGE1 = 0.7
DISTILL_LORA_STRENGTH_STAGE2 = 0.5

STAGE1_SIGMAS_FAST = (
    "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
)
STAGE2_SIGMAS_FAST = "0.85, 0.7250, 0.4219, 0.0"
STAGE2_SIGMAS_QUALITY = "0.85, 0.7933, 0.68, 0.51, 0.2833, 0.0"

QUALITY_LTXVSCHEDULER_STEPS = 50
QUALITY_LTXVSCHEDULER_MAX_SHIFT = 2.72
QUALITY_LTXVSCHEDULER_BASE_SHIFT = 0.8
QUALITY_LTXVSCHEDULER_TERMINAL = 0.0
QUALITY_CFG_STAGE1 = 3.6
QUALITY_CFG_STAGE2 = 1.0

DEFAULT_NEGATIVE_PROMPT = (
    "pc game, console game, video game, cartoon, childish, ugly, low quality, "
    "blurry, distorted, watermark, jpeg artifacts, text overlay"
)

VAE_HORIZONTAL_TILES = 2
VAE_VERTICAL_TILES = 2
VAE_TILE_OVERLAP = 6

MAX_SEED = 2**63 - 1
