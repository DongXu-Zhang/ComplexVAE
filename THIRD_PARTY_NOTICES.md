# Third-party notices

## Topology / math reference (not copied as a dependency runtime)

The encoder/decoder **channel schedule and computational graph** are informed by:

- Hybrid-SD Small VAE configuration and author `AutoencoderKL` wrapper  
  https://github.com/bytedance/Hybrid-SD (commit `dcac636c82658ee055c71b8377ee06508fb9937c`)  
  Apache-2.0
- Hugging Face Diffusers 0.27.0 `Encoder` / `Decoder` / `DiagonalGaussianDistribution` designs  
  https://github.com/huggingface/diffusers (commit `cfa7c0a93df3384685ec927c6c67d0e8a91d862a`)  
  Apache-2.0

This repository implements a **new, independent** module stack under `src/microscopy_vae/models/`.
It does **not** vendor Diffusers or Hybrid-SD source trees and does **not** load their weights.

## Other dependencies

See `pyproject.toml` for runtime dependencies (NumPy, Pydantic, PyYAML, mrcfile, tifffile, PyTorch).
Each retains its own license.
