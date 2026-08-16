"""Model factory: fresh_init only. Never accepts weight paths."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from microscopy_vae.models.initialization import assert_no_nan_params, init_module_kaiming
from microscopy_vae.models.vae import MicroscopyVAE

_FORBIDDEN_INIT_KEYS = (
    "pretrained",
    "init_from",
    "init_from_weights",
    "from_pretrained",
    "load_weights",
    "teacher",
    "checkpoint",
    "safetensors",
    "state_dict_path",
)


class ModelFactory:
    """Creates randomly initialized MicroscopyVAE instances."""

    @staticmethod
    def create_fresh(
        *,
        latent_channels: int = 4,
        encoder_block_out_channels: Sequence[int] = (128, 256, 512, 512),
        decoder_block_out_channels: Sequence[int] = (96, 192, 384, 384),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        output_activation: str = "linear",
        upsample_mode: str = "nearest",
        downsample_pad_mode: str = "asymmetric",
        downsample_preblur: bool = False,
        pretrained: Any = None,
        init_from: Any = None,
        checkpoint: Any = None,
        from_pretrained: Any = None,
        **kwargs: Any,
    ) -> MicroscopyVAE:
        for name, val in [
            ("pretrained", pretrained),
            ("init_from", init_from),
            ("checkpoint", checkpoint),
            ("from_pretrained", from_pretrained),
        ]:
            if val is not None:
                raise ValueError(
                    f"ModelFactory.create_fresh() rejects {name}={val!r}. "
                    "Use CheckpointManager.resume_exact or StageTransitionLoader instead."
                )
        for k, v in kwargs.items():
            if k in _FORBIDDEN_INIT_KEYS and v not in (None, False, "", {}):
                raise ValueError(f"ModelFactory.create_fresh() rejects forbidden kwarg {k}={v!r}")
        model = MicroscopyVAE(
            latent_channels=latent_channels,
            encoder_block_out_channels=encoder_block_out_channels,
            decoder_block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            mid_block_add_attention=mid_block_add_attention,
            output_activation=output_activation,
            upsample_mode=upsample_mode,
            downsample_pad_mode=downsample_pad_mode,
            downsample_preblur=downsample_preblur,
        )
        init_module_kaiming(model)
        assert_no_nan_params(model)
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    @staticmethod
    def from_config_dict(cfg: Dict[str, Any]) -> MicroscopyVAE:
        model_cfg = cfg.get("model", cfg)
        if not isinstance(model_cfg, dict):
            raise TypeError("model config must be a dict")
        for key in _FORBIDDEN_INIT_KEYS:
            if key in model_cfg and model_cfg[key] not in (None, False, "", {}):
                raise ValueError(f"Forbidden model init key: {key}={model_cfg[key]!r}")
        return ModelFactory.create_fresh(
            latent_channels=int(model_cfg.get("latent_channels", 4)),
            encoder_block_out_channels=tuple(
                model_cfg.get("encoder_block_out_channels", [128, 256, 512, 512])
            ),
            decoder_block_out_channels=tuple(
                model_cfg.get("decoder_block_out_channels", [96, 192, 384, 384])
            ),
            layers_per_block=int(model_cfg.get("layers_per_block", 2)),
            norm_num_groups=int(model_cfg.get("norm_num_groups", 32)),
            mid_block_add_attention=bool(model_cfg.get("mid_block_add_attention", True)),
            output_activation=str(model_cfg.get("output_activation", "linear")),
            upsample_mode=str(model_cfg.get("upsample_mode", "nearest")),
            downsample_pad_mode=str(model_cfg.get("downsample_pad_mode", "asymmetric")),
            downsample_preblur=bool(model_cfg.get("downsample_preblur", False)),
        )


def architecture_id(model: MicroscopyVAE) -> str:
    return (
        f"microvae_f{model.spatial_compression}_z{model.latent_channels}"
        f"_enc{'-'.join(map(str, model.encoder.block_out_channels))}"
        f"_dec{'-'.join(map(str, model.decoder.block_out_channels))}"
    )
