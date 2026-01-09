"""Smoke test for WorldModel with new action conditioning features."""

import torch
from model.src.models.world_model import WorldModel


def test_world_model_initialization():
    """Test that WorldModel can be initialized with action encoder."""
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=True,
        action_embed_dim=12,
        contrastive_weight=0.1,
        contrastive_margin=1.0,
        grad_detach_schedule_k=4,
    )
    print("✓ WorldModel initialized successfully with action encoder")
    
    # Check that action encoder exists
    assert hasattr(model, 'action_encoder')
    assert model.action_encoder is not None
    assert model.use_action_encoder is True
    print("✓ ActionEncoder is present")
    
    # Check contrastive loss
    assert hasattr(model, 'contrastive_loss_fn')
    print("✓ ContrastiveActionLoss is present")


def test_world_model_forward():
    """Test WorldModel forward pass with action encoder."""
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=True,
        action_embed_dim=12,
        contrastive_weight=0.1,
    )
    model.eval()
    
    batch_size = 2
    seq_len = 4
    images = torch.randn(batch_size, seq_len, 3, 64, 64)
    actions = torch.randn(batch_size, seq_len - 1, 6)
    
    # Forward pass
    output = model(images, actions)
    
    # Check output structure
    assert hasattr(output, 'loss')
    assert hasattr(output, 'rec_loss')
    assert hasattr(output, 'kld')
    assert hasattr(output, 'contrastive_loss')
    assert hasattr(output, 'action_sensitivity')
    assert hasattr(output, 'latent_action_variance')
    print("✓ Forward pass successful with all expected outputs")
    
    # Check tensor shapes
    assert output.x_rec.shape == images.shape
    print("✓ Reconstruction shape matches input")
    
    # Check that values are finite
    assert torch.isfinite(output.loss).all()
    assert torch.isfinite(output.rec_loss).all()
    assert torch.isfinite(output.kld).all()
    print("✓ All losses are finite")


def test_world_model_without_action_encoder():
    """Test WorldModel can work without action encoder (backward compatibility)."""
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=False,
    )
    model.eval()
    
    batch_size = 2
    seq_len = 4
    images = torch.randn(batch_size, seq_len, 3, 64, 64)
    actions = torch.randn(batch_size, seq_len - 1, 6)
    
    # Forward pass
    output = model(images, actions)
    
    assert torch.isfinite(output.loss).all()
    print("✓ Model works without action encoder (backward compatible)")


def test_world_model_gradient_flow():
    """Test that gradients flow through the model."""
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=True,
        contrastive_weight=0.1,
    )
    model.train()
    
    batch_size = 2
    seq_len = 4
    images = torch.randn(batch_size, seq_len, 3, 64, 64)
    actions = torch.randn(batch_size, seq_len - 1, 6)
    
    # Forward pass
    output = model(images, actions)
    
    # Backward pass
    output.loss.backward()
    
    # Check that gradients exist for key parameters
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    
    assert has_grad, "No gradients computed"
    print("✓ Gradients flow through the model")


def test_world_model_imagine():
    """Test the imagine method with action encoder."""
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=True,
    )
    model.eval()
    
    start_image = torch.randn(1, 3, 64, 64)
    actions = torch.randn(5, 6)  # 5 actions -> 6 images
    
    with torch.no_grad():
        imagined = model.imagine(start_image, actions)
    
    assert imagined.shape == (6, 3, 64, 64)
    assert torch.isfinite(imagined).all()
    print("✓ Imagine method works with action encoder")


if __name__ == "__main__":
    print("Running WorldModel smoke tests...\n")
    test_world_model_initialization()
    test_world_model_forward()
    test_world_model_without_action_encoder()
    test_world_model_gradient_flow()
    test_world_model_imagine()
    print("\n✅ All WorldModel smoke tests passed!")
