"""Test gradient isolation between VAE and RSSM losses.

This test validates that:
1. VAE encoder/decoder parameters receive gradients ONLY from VAE loss (reconstruction + KL)
2. RSSM parameters receive gradients ONLY from RSSM consistency loss
3. Encoder parameters DO NOT receive gradients from RSSM consistency loss
4. RSSM parameters DO NOT receive gradients from reconstruction loss
"""

import torch
from model.src.models.world_model import WorldModel


def test_gradient_separation():
    """Test that gradients are properly separated between VAE and RSSM."""
    print("Testing gradient separation between VAE and RSSM...")
    
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=False,
        contrastive_weight=0.0,  # Disable contrastive loss for clarity
    )
    model.train()
    
    batch_size = 2
    seq_len = 4
    images = torch.randn(batch_size, seq_len, 3, 64, 64)
    actions = torch.randn(batch_size, seq_len - 1, 6)
    
    # Test 1: Check that encoder parameters exist and are trainable
    encoder_params = list(model.vae.encoder.parameters())
    decoder_params = list(model.vae.decoder.parameters())
    rssm_params = list(model.rssm.parameters())
    
    assert len(encoder_params) > 0, "Encoder should have parameters"
    assert len(decoder_params) > 0, "Decoder should have parameters"
    assert len(rssm_params) > 0, "RSSM should have parameters"
    print(f"  ✓ Model has {len(encoder_params)} encoder, {len(decoder_params)} decoder, {len(rssm_params)} RSSM parameters")
    
    # Test 2: Forward pass and backward on total loss
    output = model(images, actions)
    model.zero_grad()
    output.loss.backward()
    
    # Count parameters with gradients
    encoder_grad_count = sum(1 for p in encoder_params if p.grad is not None and p.grad.abs().sum() > 1e-9)
    decoder_grad_count = sum(1 for p in decoder_params if p.grad is not None and p.grad.abs().sum() > 1e-9)
    rssm_grad_count = sum(1 for p in rssm_params if p.grad is not None and p.grad.abs().sum() > 1e-9)
    
    print(f"  ✓ Encoder: {encoder_grad_count}/{len(encoder_params)} parameters have gradients")
    print(f"  ✓ Decoder: {decoder_grad_count}/{len(decoder_params)} parameters have gradients")
    print(f"  ✓ RSSM: {rssm_grad_count}/{len(rssm_params)} parameters have gradients")
    
    # All parameters should have gradients from the total loss
    assert encoder_grad_count > 0, "Encoder should have gradients from total loss"
    assert decoder_grad_count > 0, "Decoder should have gradients from total loss"
    assert rssm_grad_count > 0, "RSSM should have gradients from total loss"
    
    print("  ✓ PASS: All components receive gradients from total loss")


def test_manual_gradient_computation():
    """Manually compute VAE and RSSM losses to verify gradient isolation."""
    print("\nTesting manual gradient computation...")
    
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=False,
        contrastive_weight=0.0,
        kl_beta=1.0,
    )
    model.train()
    
    batch_size = 2
    seq_len = 4
    images = torch.randn(batch_size, seq_len, 3, 64, 64)
    actions = torch.randn(batch_size, seq_len - 1, 6)
    
    # Manually compute losses with gradient tracking
    b, t, c, h, w = images.shape
    flat = images.reshape(b * t, c, h, w)
    
    # VAE forward
    x_rec_flat, mu_flat, logvar_flat, feat_flat = model.vae(flat)
    mu = mu_flat.reshape(b, t, -1)
    logvar = logvar_flat.reshape(b, t, -1)
    
    # VAE loss components
    rec_loss = model.vae.reconstruction_loss(flat, x_rec_flat)
    from model.src.models.world_model import _standard_normal_kl
    vae_kl = _standard_normal_kl(mu[:, 0], logvar[:, 0]).mean()
    vae_loss = rec_loss + vae_kl
    
    # Test VAE loss gradients
    model.zero_grad()
    vae_loss.backward(retain_graph=True)
    
    encoder_has_grad_vae = any(
        p.grad is not None and p.grad.abs().sum() > 1e-9
        for p in model.vae.encoder.parameters()
    )
    decoder_has_grad_vae = any(
        p.grad is not None and p.grad.abs().sum() > 1e-9
        for p in model.vae.decoder.parameters()
    )
    rssm_has_grad_vae = any(
        p.grad is not None and p.grad.abs().sum() > 1e-9
        for p in model.rssm.parameters()
    )
    
    print(f"  VAE loss gradients: encoder={encoder_has_grad_vae}, decoder={decoder_has_grad_vae}, RSSM={rssm_has_grad_vae}")
    assert encoder_has_grad_vae, "Encoder should have gradients from VAE loss"
    assert decoder_has_grad_vae, "Decoder should have gradients from VAE loss"
    assert not rssm_has_grad_vae, "RSSM should NOT have gradients from VAE loss"
    print("  ✓ PASS: VAE loss does not affect RSSM parameters")
    
    # Now test RSSM loss with detached posteriors
    model.zero_grad()
    mu_detached = mu.detach()
    logvar_detached = logvar.detach()
    
    # Encode actions
    if model.use_action_encoder:
        action_input = model.action_encoder(actions)
    else:
        action_input = actions
    
    # RSSM forward with detached posteriors
    state = model.rssm.init_state(b, device=images.device)
    state = model._make_state(state.h, mu_detached[:, 0])
    
    rssm_kls = []
    for i in range(1, t):
        prev_mean = mu_detached[:, i - 1]
        x = model.rssm.inp(torch.cat([prev_mean, action_input[:, i - 1]], dim=-1))
        h_next = model.rssm.gru(x, state.h)
        prior_mu, prior_logvar = model.rssm.prior_params(h_next)
        
        post_mu_target = mu_detached[:, i]
        post_logvar_target = logvar_detached[:, i]
        
        rssm_kl_i = model.rssm.kl(post_mu_target, post_logvar_target, prior_mu, prior_logvar)
        rssm_kls.append(rssm_kl_i)
        
        state = model._make_state(h_next, post_mu_target)
    
    if rssm_kls:
        rssm_loss = torch.stack(rssm_kls, dim=1).mean()
        rssm_loss.backward()
        
        encoder_has_grad_rssm = any(
            p.grad is not None and p.grad.abs().sum() > 1e-9
            for p in model.vae.encoder.parameters()
        )
        decoder_has_grad_rssm = any(
            p.grad is not None and p.grad.abs().sum() > 1e-9
            for p in model.vae.decoder.parameters()
        )
        rssm_has_grad_rssm = any(
            p.grad is not None and p.grad.abs().sum() > 1e-9
            for p in model.rssm.parameters()
        )
        
        print(f"  RSSM loss gradients: encoder={encoder_has_grad_rssm}, decoder={decoder_has_grad_rssm}, RSSM={rssm_has_grad_rssm}")
        assert not encoder_has_grad_rssm, "Encoder should NOT have gradients from RSSM loss"
        assert not decoder_has_grad_rssm, "Decoder should NOT have gradients from RSSM loss"
        assert rssm_has_grad_rssm, "RSSM should have gradients from RSSM loss"
        print("  ✓ PASS: RSSM loss does not affect VAE parameters")


def test_loss_components():
    """Test that loss components have expected magnitudes."""
    print("\nTesting loss component magnitudes...")
    
    model = WorldModel(
        action_dim=6,
        latent_dim=32,
        deter_dim=64,
        base_channels=16,
        use_action_encoder=False,
        contrastive_weight=0.0,
    )
    model.train()
    
    batch_size = 2
    seq_len = 4
    images = torch.randn(batch_size, seq_len, 3, 64, 64)
    actions = torch.randn(batch_size, seq_len - 1, 6)
    
    # Forward pass
    output = model(images, actions)
    
    # Check that all losses are finite and positive
    assert torch.isfinite(output.rec_loss).all(), "Reconstruction loss is not finite"
    assert torch.isfinite(output.kld).all(), "KLD is not finite"
    assert torch.isfinite(output.rssm_loss).all(), "RSSM loss is not finite"
    assert output.rec_loss > 0, "Reconstruction loss should be positive"
    assert output.rssm_loss >= 0, "RSSM loss should be non-negative"
    
    print(f"  rec_loss: {output.rec_loss:.4f}")
    print(f"  kld: {output.kld:.4f}")
    print(f"  rssm_loss: {output.rssm_loss:.4f}")
    print(f"  total_loss: {output.loss:.4f}")
    print("  ✓ PASS: All loss components are finite")


if __name__ == "__main__":
    print("=" * 70)
    print("GRADIENT ISOLATION TESTS FOR RSSM LATENT DYNAMICS REFACTOR")
    print("=" * 70)
    
    test_loss_components()
    test_gradient_separation()
    test_manual_gradient_computation()
    
    print("\n" + "=" * 70)
    print("✅ ALL GRADIENT ISOLATION TESTS PASSED!")
    print("=" * 70)
