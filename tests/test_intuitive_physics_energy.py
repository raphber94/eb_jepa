"""Unit tests for clip_energy() and clip_pixel_energy() in eval.py."""
import pytest
import torch

from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.training_utils import load_config
from examples.intuitive_physics.eval import (
    _sanity_check,
    clip_energy,
    clip_pixel_energy,
)
from examples.intuitive_physics.main import build_jepa


@pytest.fixture
def small_jepa():
    cfg = load_config("examples/intuitive_physics/cfgs/smoke.yaml")
    device = torch.device("cpu")
    jepa = build_jepa(cfg, device)
    jepa.eval()
    return jepa, cfg, device


def test_clip_energy_shape(small_jepa):
    jepa, cfg, device = small_jepa
    clips = torch.rand(6, 1, cfg.data.T, 64, 64)
    energies = clip_energy(jepa, clips, nsteps=cfg.model.steps, device=device, batch_size=4)
    assert energies.shape == (6,), f"Expected [6], got {energies.shape}"
    assert (energies >= 0).all(), "Energies should be non-negative"


def test_clip_energy_sanity(small_jepa):
    """Per-clip mean must match the batch-mean from jepa.unroll(compute_loss=True)."""
    jepa, cfg, device = small_jepa
    clips = torch.rand(8, 1, cfg.data.T, 64, 64)
    _sanity_check(jepa, clips, cfg.model.steps, device, batch_size=8, tol=1e-3)


def test_clip_pixel_energy_shape(small_jepa):
    jepa, cfg, device = small_jepa
    clips = torch.rand(6, 1, cfg.data.T, 64, 64)
    decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs, hidden_dim=cfg.model.dstc)
    decoder.eval()
    energies = clip_pixel_energy(decoder, jepa, clips, device=device, batch_size=4)
    assert energies.shape == (6,), f"Expected [6], got {energies.shape}"
    assert (energies >= 0).all()


def test_different_clips_give_different_energies(small_jepa):
    jepa, cfg, device = small_jepa
    clips_a = torch.rand(4, 1, cfg.data.T, 64, 64)
    clips_b = torch.rand(4, 1, cfg.data.T, 64, 64)
    e_a = clip_energy(jepa, clips_a, cfg.model.steps, device)
    e_b = clip_energy(jepa, clips_b, cfg.model.steps, device)
    assert not torch.allclose(e_a, e_b), "Different clips should give different energies"
