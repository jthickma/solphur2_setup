# ComfyUI-Sulphur2

Native ComfyUI custom nodes for the Sulphur-2 two-stage text-to-video recipe
used by this repository's HTTP API.

## Nodes

- `Sulphur2Generate`: one-node text-to-video path.
- `Sulphur2BaseLoader`: loads the Sulphur FP8 checkpoint, Gemma3 text encoder,
  audio VAE, latent upscaler, and applies the Blackwell SageAttention override.
- `Sulphur2Conditioning`: prompt and negative prompt conditioning with LTX frame-rate metadata.
- `Sulphur2Stage1Sampler`: half-resolution base diffusion, with `quality` and `fast` modes.
- `Sulphur2SpatialUpscale`: separates AV latents, upscales video only, and exposes both streams.
- `Sulphur2Stage2Sampler`: final refinement with the canonical cond_safe distill LoRA at 0.5.
- `Sulphur2DecodeAndSave`: tiled VAE decode, audio decode, video mux, and MP4 save.
- `Sulphur2LatentPrep`, `Sulphur2TextEncoder`, `Sulphur2DistillLoRALoader`,
  and `Sulphur2PromptEnhancer` are utility nodes for power users.

## Workflows

- `workflows/ltx23_t2v_sulphur2_base.json`: quality mode, equivalent to the default
  `POST /generate` path.
- `workflows/ltx23_t2v_sulphur2_fast.json`: fast distilled preview mode.
- `workflows/ltx23_t2v_sulphur2_one_node.json`: compact one-node workflow.

The mid-level workflow carries `AUDIO_VAE` and `LATENT_UPSCALE_MODEL` as
visible graph outputs from `Sulphur2BaseLoader`. That keeps the ComfyUI graph
cacheable and avoids hidden file loads in sampler nodes.

## Install

Inside a ComfyUI checkout:

```bash
cd custom_nodes
git clone <this repo or extracted ComfyUI-Sulphur2 directory>
cd ComfyUI-Sulphur2
pip install -r requirements.txt
```

This repository's Docker image installs the node pack automatically.

## Dependencies

The pack expects the same pinned dependencies as the Docker deployment:

- ComfyUI v0.21.0
- ComfyUI-LTXVideo at the pinned SHA in `../versions.env`
- ComfyUI-KJNodes at the pinned SHA in `../versions.env`
- ComfyMath at the pinned SHA in `../versions.env`
- The Sulphur, Gemma3, distill LoRA, and latent upscaler model files downloaded
  by `scripts/download_models.sh`

The `quality` default is the validated 1280x704, 241-frame envelope. Larger or
longer requests are allowed up to the LTX-2.3 ceiling, but the node pack logs a
warning because 1080p x 20s quality mode is a known audio-mux risk for this
Sulphur fine-tune.
