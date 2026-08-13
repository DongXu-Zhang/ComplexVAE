# Architecture

See Package A for full decisions. Summary:

- **S1:** HQ → VAE → HQ reconstruction (pure codec).
- **Model:** independent `MicroscopyVAE` with asymmetric Enc/Dec channels, f8, z=4, linear output.
- **System:** `HQCodecSystem` owns VAE; no `restore_lr`.
- **Task:** `HQCodecTask` + `HQCodecLossComposer`.
- **LatentSpec:** versioned export affine; never SD `0.18215` as training constant.
