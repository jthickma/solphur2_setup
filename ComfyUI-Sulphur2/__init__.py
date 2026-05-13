from .nodes.sulphur2_base import Sulphur2BaseLoader, Sulphur2TextEncoder
from .nodes.sulphur2_conditioning import Sulphur2Conditioning
from .nodes.sulphur2_decode import Sulphur2DecodeAndSave
from .nodes.sulphur2_distill import Sulphur2DistillLoRALoader
from .nodes.sulphur2_generate import Sulphur2Generate
from .nodes.sulphur2_latent_prep import Sulphur2LatentPrep
from .nodes.sulphur2_prompt_enhancer import Sulphur2PromptEnhancer
from .nodes.sulphur2_sampler import Sulphur2Stage1Sampler, Sulphur2Stage2Sampler
from .nodes.sulphur2_spatial import Sulphur2SpatialUpscale

NODE_CLASS_MAPPINGS = {
    "Sulphur2BaseLoader": Sulphur2BaseLoader,
    "Sulphur2TextEncoder": Sulphur2TextEncoder,
    "Sulphur2Conditioning": Sulphur2Conditioning,
    "Sulphur2Stage1Sampler": Sulphur2Stage1Sampler,
    "Sulphur2SpatialUpscale": Sulphur2SpatialUpscale,
    "Sulphur2Stage2Sampler": Sulphur2Stage2Sampler,
    "Sulphur2DecodeAndSave": Sulphur2DecodeAndSave,
    "Sulphur2DistillLoRALoader": Sulphur2DistillLoRALoader,
    "Sulphur2LatentPrep": Sulphur2LatentPrep,
    "Sulphur2PromptEnhancer": Sulphur2PromptEnhancer,
    "Sulphur2Generate": Sulphur2Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sulphur2BaseLoader": "Sulphur 2 Load Checkpoint (Blackwell)",
    "Sulphur2TextEncoder": "Sulphur 2 Text Encoder (Gemma3)",
    "Sulphur2Conditioning": "Sulphur 2 Conditioning",
    "Sulphur2Stage1Sampler": "Sulphur 2 Stage 1 Sampler",
    "Sulphur2SpatialUpscale": "Sulphur 2 Spatial Upscale x2",
    "Sulphur2Stage2Sampler": "Sulphur 2 Stage 2 Sampler",
    "Sulphur2DecodeAndSave": "Sulphur 2 Decode and Save MP4",
    "Sulphur2DistillLoRALoader": "Sulphur 2 Distill LoRA Loader",
    "Sulphur2LatentPrep": "Sulphur 2 Latent Prep",
    "Sulphur2PromptEnhancer": "Sulphur 2 Prompt Enhancer",
    "Sulphur2Generate": "Sulphur 2 Generate (T2V)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
