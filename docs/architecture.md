# Architecture

See Package A for full decisions. Summary:

- **S1:** HQ → VAE → HQ reconstruction (pure codec).
- **Model:** independent `MicroscopyVAE`. Spatial compression is `2**(n_stages-1)` from `encoder_block_out_channels` length (must match decoder). f4 = 3 stages → `4×64×64`; f8 = 4 stages `[128,256,512,512]`/`[96,192,384,384]` → `4×32×32`. z=4, linear output, bottleneck self-attention. Not an interpolation of f4 latent.
- **System:** `HQCodecSystem` owns VAE; no `restore_lr`.
- **Task:** `HQCodecTask` + `HQCodecLossComposer`.
- **LatentSpec:** versioned export affine; never SD `0.18215` as training constant.
