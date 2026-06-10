from __future__ import annotations

from torch import nn

from xa202608.models.ad_tcn_mscdim import AdTcnMscdimRegressor
from xa202608.models.esn import LeakyEsnRegressor
from xa202608.models.recurrent import RecurrentRulRegressor
from xa202608.models.tcn import PSaMcdTcnRegressor, PTcnRegressor
from xa202608.models.transformer import TransformerRulRegressor


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    if name in {"transformer", "transformer_rul", "rul_transformer"}:
        return TransformerRulRegressor(
            input_channels=int(model_cfg["input_channels"]),
            d_model=int(model_cfg.get("d_model", model_cfg.get("hidden_channels", 128))),
            num_heads=int(model_cfg.get("num_heads", model_cfg.get("attention_heads", 4))),
            num_layers=int(model_cfg.get("num_layers", model_cfg.get("num_encoder_blocks", 2))),
            dim_feedforward=int(model_cfg.get("dim_feedforward", model_cfg.get("ff_dim", 256))),
            dropout=float(model_cfg.get("dropout", 0.1)),
            output_dim=int(model_cfg.get("output_dim", 1)),
            max_len=int(model_cfg.get("max_len", 512)),
            pooling=str(model_cfg.get("pooling", "last")),
            head_hidden_dims=[int(x) for x in model_cfg.get("head_hidden_dims", [])] or None,
        )
    if name in {"leaky_esn", "esn", "leakyesn"}:
        return LeakyEsnRegressor(
            input_channels=int(model_cfg["input_channels"]),
            reservoir_size=int(model_cfg.get("reservoir_size", model_cfg.get("hidden_channels", 256))),
            spectral_radius=float(model_cfg.get("spectral_radius", 0.9)),
            leak_rate=float(model_cfg.get("leak_rate", 0.3)),
            input_scale=float(model_cfg.get("input_scale", 0.5)),
            reservoir_density=float(model_cfg.get("reservoir_density", 0.1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            output_dim=int(model_cfg.get("output_dim", 1)),
            train_reservoir=bool(model_cfg.get("train_reservoir", False)),
            head_hidden_dims=[int(x) for x in model_cfg.get("head_hidden_dims", [])] or None,
        )
    if name in {"lstm", "gru", "recurrent", "rnn"}:
        cell_type = str(model_cfg.get("cell_type", name if name in {"lstm", "gru"} else "lstm")).lower()
        return RecurrentRulRegressor(
            input_channels=int(model_cfg["input_channels"]),
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            output_dim=int(model_cfg.get("output_dim", 1)),
            cell_type=cell_type,
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            pooling=str(model_cfg.get("pooling", "last")),
            use_stage_head=bool(model_cfg.get("use_stage_head", False)),
            num_stages=int(model_cfg.get("num_stages", 3)),
            head_hidden_dims=[int(x) for x in model_cfg.get("head_hidden_dims", [])] or None,
        )
    common = {
        "input_channels": int(model_cfg["input_channels"]),
        "hidden_channels": int(model_cfg["hidden_channels"]),
        "kernel_size": int(model_cfg["kernel_size"]),
        "dilations": [int(x) for x in model_cfg["dilations"]],
        "dropout": float(model_cfg["dropout"]),
        "output_dim": int(model_cfg.get("output_dim", 1)),
        "use_batch_norm": bool(model_cfg.get("use_batch_norm", False)),
        "head_hidden_dims": [int(x) for x in model_cfg.get("head_hidden_dims", [])] or None,
    }
    if name in {"p_tcn", "ptcn"}:
        return PTcnRegressor(**common)
    if name in {"ad_tcn_mscdim", "adtcn_mscdim", "ad_tcn_m"}:
        common.pop("use_batch_norm", None)
        common.update(
            {
                "window_size": int(config["data"]["window_size"]),
                "num_ad_blocks": int(model_cfg.get("num_ad_blocks", 2)),
                "mscdim_kernel_sizes": [int(x) for x in model_cfg.get("mscdim_kernel_sizes", [1, 3, 5])],
                "use_weight_norm": bool(model_cfg.get("use_weight_norm", True)),
                "use_stage_head": bool(model_cfg.get("use_stage_head", False)),
                "num_stages": int(model_cfg.get("num_stages", 3)),
            }
        )
        return AdTcnMscdimRegressor(**common)
    if name in {"p_sa_mcd_tcn", "psa_mcd_tcn"}:
        common.update(
            {
                "attention_heads": int(model_cfg.get("attention_heads", 4)),
                "use_temporal_attention": bool(model_cfg.get("use_temporal_attention", False)),
                "use_channel_attention": bool(model_cfg.get("use_channel_attention", False)),
                "use_temporal_descriptors": bool(model_cfg.get("use_temporal_descriptors", False)),
                "use_stage_head": bool(model_cfg.get("use_stage_head", True)),
                "num_stages": int(model_cfg.get("num_stages", 3)),
            }
        )
        return PSaMcdTcnRegressor(**common)
    raise ValueError(f"unknown model name: {model_cfg['name']}")
