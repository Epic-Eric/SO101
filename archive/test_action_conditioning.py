"""Tests for action encoder and contrastive loss modules."""

import torch
from model.src.models.action_encoder import ActionEncoder
from model.src.models.contrastive_loss import ContrastiveActionLoss


def test_action_encoder_init():
    """Test ActionEncoder initialization."""
    encoder = ActionEncoder(action_dim=6, embed_dim=12, hidden_dim=24, num_layers=2)
    assert encoder.action_dim == 6
    assert encoder.embed_dim == 12
    assert encoder.hidden_dim == 24
    assert encoder.num_layers == 2


def test_action_encoder_forward():
    """Test ActionEncoder forward pass."""
    encoder = ActionEncoder(action_dim=6, embed_dim=12)
    batch_size = 4
    action = torch.randn(batch_size, 6)
    
    embedding = encoder(action)
    assert embedding.shape == (batch_size, 12)
    assert not torch.isnan(embedding).any()
    assert not torch.isinf(embedding).any()


def test_action_encoder_default_params():
    """Test ActionEncoder with default parameters."""
    encoder = ActionEncoder(action_dim=6)
    action = torch.randn(4, 6)
    
    # Default embed_dim should be same as action_dim
    embedding = encoder(action)
    assert embedding.shape == (4, 6)


def test_action_encoder_batch_independence():
    """Test that ActionEncoder processes each batch element independently."""
    encoder = ActionEncoder(action_dim=6, embed_dim=12)
    action1 = torch.randn(1, 6)
    action2 = torch.randn(1, 6)
    action_batch = torch.cat([action1, action2], dim=0)
    
    # Individual processing
    embed1 = encoder(action1)
    embed2 = encoder(action2)
    
    # Batch processing
    embed_batch = encoder(action_batch)
    
    # Should be the same
    assert torch.allclose(embed_batch[0], embed1[0], atol=1e-6)
    assert torch.allclose(embed_batch[1], embed2[0], atol=1e-6)


def test_contrastive_loss_init():
    """Test ContrastiveActionLoss initialization."""
    loss_fn = ContrastiveActionLoss(margin=1.0, distance_type="l2")
    assert loss_fn.margin == 1.0
    assert loss_fn.distance_type == "l2"


def test_contrastive_loss_l2_distance():
    """Test L2 distance computation."""
    loss_fn = ContrastiveActionLoss(distance_type="l2")
    z1 = torch.tensor([[1.0, 0.0, 0.0]])
    z2 = torch.tensor([[0.0, 1.0, 0.0]])
    
    distance = loss_fn.compute_distance(z1, z2)
    expected = torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(distance, expected, atol=1e-6)


def test_contrastive_loss_cosine_distance():
    """Test cosine distance computation."""
    loss_fn = ContrastiveActionLoss(distance_type="cosine")
    # Orthogonal vectors should have cosine similarity of 0, distance of 1
    z1 = torch.tensor([[1.0, 0.0, 0.0]])
    z2 = torch.tensor([[0.0, 1.0, 0.0]])
    
    distance = loss_fn.compute_distance(z1, z2)
    assert torch.allclose(distance, torch.tensor(1.0), atol=1e-6)


def test_contrastive_loss_pairwise():
    """Test pairwise contrastive loss."""
    loss_fn = ContrastiveActionLoss(margin=1.0, distance_type="l2")
    
    # Same latent states (distance=0, should incur loss)
    z_same = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    loss_same = loss_fn.pairwise_loss(z_same[0:1], z_same[1:2])
    assert loss_same > 0  # Should incur loss because distance < margin
    
    # Very different latent states (distance > margin, no loss)
    z_diff1 = torch.tensor([[1.0, 2.0, 3.0]])
    z_diff2 = torch.tensor([[10.0, 20.0, 30.0]])
    loss_diff = loss_fn.pairwise_loss(z_diff1, z_diff2)
    assert torch.allclose(loss_diff, torch.tensor(0.0), atol=1e-6)


def test_contrastive_loss_triplet():
    """Test triplet-style contrastive loss."""
    loss_fn = ContrastiveActionLoss(margin=1.0, distance_type="l2")
    
    # Anchor
    z_anchor = torch.tensor([[0.0, 0.0, 0.0]])
    # Positive (close to anchor)
    z_pos = torch.tensor([[0.1, 0.1, 0.1]])
    # Negative (far from anchor)
    z_neg = torch.tensor([[5.0, 5.0, 5.0]])
    
    loss = loss_fn(z_anchor, z_pos, z_neg)
    # Positive is close, negative is far, so loss should be near 0
    assert loss >= 0  # Loss should always be non-negative


def test_action_encoder_gradient_flow():
    """Test that gradients flow through ActionEncoder."""
    encoder = ActionEncoder(action_dim=6, embed_dim=12)
    action = torch.randn(4, 6, requires_grad=True)
    
    embedding = encoder(action)
    loss = embedding.sum()
    loss.backward()
    
    # Check that gradients exist
    assert action.grad is not None
    assert not torch.isnan(action.grad).any()


if __name__ == "__main__":
    # Run tests
    test_action_encoder_init()
    test_action_encoder_forward()
    test_action_encoder_default_params()
    test_action_encoder_batch_independence()
    test_contrastive_loss_init()
    test_contrastive_loss_l2_distance()
    test_contrastive_loss_cosine_distance()
    test_contrastive_loss_pairwise()
    test_contrastive_loss_triplet()
    test_action_encoder_gradient_flow()
    print("All tests passed!")
