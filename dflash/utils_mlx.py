import json
from pathlib import Path
from typing import Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download

from .model_mlx import DFlashConfig, DFlashDraftModel


def load(model_id: str, use_paroquant: bool = False) -> Tuple:
    """Load target model from HuggingFace.

    Args:
        model_id: HuggingFace model ID or local path
        use_paroquant: Use ParoQuant backend (for PARO models)

    Returns:
        Tuple of (model, tokenizer)
    """
    if use_paroquant:
        from paroquant.inference.backends.mlx.load import load as paro_load
        model, tokenizer, _ = paro_load(model_id, force_text=True)
    else:
        from mlx_lm import load as mlx_lm_load
        model, tokenizer = mlx_lm_load(model_id)

    return model, tokenizer


def load_draft(draft_id: str) -> DFlashDraftModel:
    """Load draft model from HuggingFace.

    Args:
        draft_id: HuggingFace model ID or local path

    Returns:
        DFlashDraftModel instance
    """
    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    cfg = json.loads((path / "config.json").read_text())

    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"]["mask_token_id"],
    )

    weights = {k: v for f in path.glob("*.safetensors") for k, v in mx.load(str(f)).items()}
    model = DFlashDraftModel(config)
    model.load_weights(list(weights.items()))

    return model
