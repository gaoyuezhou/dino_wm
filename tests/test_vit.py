"""Regression tests for the causal predictor attention mask."""

import torch

from models.vit import ViTPredictor


def _tiny_predictor():
    return ViTPredictor(
        num_patches=2,
        num_frames=3,
        dim=4,
        depth=1,
        heads=1,
        dim_head=4,
        mlp_dim=8,
        pool="mean",
    )


def test_predictor_mask_follows_module_device_and_runs_on_cpu():
    predictor = _tiny_predictor()
    attention = predictor.transformer.layers[0][0]

    assert attention.bias.device == torch.device("cpu")
    inputs = torch.randn(2, 6, 4)
    outputs = predictor(inputs)
    assert outputs.shape == inputs.shape


def test_predictor_mask_is_not_serialized():
    """Old checkpoints must load without a newly introduced mask key."""
    predictor = _tiny_predictor()
    assert "transformer.layers.0.0.bias" not in predictor.state_dict()
