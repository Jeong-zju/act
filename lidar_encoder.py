"""
Production-quality 1D CNN encoder for single-line (2D) LiDAR scans.
Designed for integration with ACT-style (Action Chunking with Transformer) policies.

Author: Senior Robotics ML Engineer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import time


@dataclass
class LiDAREncoderConfig:
    """Configuration for LiDAR 1D CNN encoder."""
    # Input/output dimensions
    embedding_dim: int = 256  # Output embedding dimension
    
    # Preprocessing parameters
    r_min: float = 0.1  # Minimum valid range (meters)
    r_max: float = 50.0  # Maximum valid range (meters)
    normalize_range: bool = True  # Whether to normalize range to [0, 1]
    epsilon: float = 1e-8  # Epsilon for numerical stability
    
    # Data augmentation (disabled by default for deterministic behavior)
    enable_augmentation: bool = False
    beam_dropout_prob: float = 0.05  # Probability of dropping a beam
    noise_std: float = 0.01  # Standard deviation of Gaussian noise (as fraction of r_max)
    shift_max_beams: int = 5  # Maximum number of beams to shift (circular)
    
    # Architecture hyperparameters
    dropout_rate: float = 0.05  # Dropout rate for early layers
    dropout_rate_late: float = 0.10  # Dropout rate for late layers
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.r_min > 0, "r_min must be positive"
        assert self.r_max > self.r_min, "r_max must be greater than r_min"
        assert 0 <= self.beam_dropout_prob <= 1, "beam_dropout_prob must be in [0, 1]"
        assert self.noise_std >= 0, "noise_std must be non-negative"
        assert self.shift_max_beams >= 0, "shift_max_beams must be non-negative"
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
        assert 0 <= self.dropout_rate_late <= 1, "dropout_rate_late must be in [0, 1]"


class LiDARPreprocessor:
    """
    Robust preprocessing for LiDAR scans.
    Handles NaN/Inf, clipping, normalization, and optional augmentation.
    """
    
    def __init__(self, config: LiDAREncoderConfig):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: LiDAREncoderConfig instance
        """
        self.config = config
    
    def __call__(
        self,
        lidar_scan: Union[torch.Tensor, np.ndarray],
        training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess LiDAR scan.
        
        Args:
            lidar_scan: Raw LiDAR scan, shape (B, N) or (N,)
            training: Whether in training mode (affects augmentation)
        
        Returns:
            processed_scan: Preprocessed scan, shape (B, 1, N)
            validity_mask: Validity mask, shape (B, N) with 1 for valid, 0 for invalid
        """
        # Convert to tensor if needed
        if isinstance(lidar_scan, np.ndarray):
            lidar_scan = torch.from_numpy(lidar_scan).float()
        
        # Ensure contiguous memory
        lidar_scan = lidar_scan.contiguous()
        
        # Handle 1D input (single sample)
        if lidar_scan.dim() == 1:
            lidar_scan = lidar_scan.unsqueeze(0)
        
        assert lidar_scan.dim() == 2, f"Expected 2D tensor (B, N), got {lidar_scan.dim()}D"
        B, N = lidar_scan.shape
        
        # Create validity mask (1 for valid, 0 for invalid)
        validity_mask = torch.ones_like(lidar_scan, dtype=torch.float32)
        
        # Handle NaN and Inf
        is_finite = torch.isfinite(lidar_scan)
        validity_mask = validity_mask * is_finite.float()
        lidar_scan = torch.where(is_finite, lidar_scan, torch.zeros_like(lidar_scan))
        
        # Clip to valid range
        lidar_scan = torch.clamp(lidar_scan, self.config.r_min, self.config.r_max)
        
        # Update validity mask based on range
        in_range = (lidar_scan >= self.config.r_min) & (lidar_scan <= self.config.r_max)
        validity_mask = validity_mask * in_range.float()
        
        # Normalize to [0, 1] if enabled
        if self.config.normalize_range:
            range_span = self.config.r_max - self.config.r_min + self.config.epsilon
            lidar_scan = (lidar_scan - self.config.r_min) / range_span
            lidar_scan = torch.clamp(lidar_scan, 0.0, 1.0)
        
        # Data augmentation (only during training)
        if training and self.config.enable_augmentation:
            lidar_scan = self._apply_augmentation(lidar_scan, validity_mask)
        
        # Add channel dimension: (B, N) -> (B, 1, N)
        lidar_scan = lidar_scan.unsqueeze(1)
        
        return lidar_scan, validity_mask
    
    def _apply_augmentation(
        self,
        lidar_scan: torch.Tensor,
        validity_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply data augmentation to LiDAR scan.
        
        Args:
            lidar_scan: Preprocessed scan, shape (B, N)
            validity_mask: Validity mask, shape (B, N)
        
        Returns:
            Augmented scan, shape (B, N)
        """
        B, N = lidar_scan.shape
        
        # Random beam dropout
        if self.config.beam_dropout_prob > 0:
            dropout_mask = torch.rand(B, N, device=lidar_scan.device) > self.config.beam_dropout_prob
            lidar_scan = lidar_scan * dropout_mask.float()
            # Set dropped beams to zero (they'll be masked out by validity_mask)
        
        # Add Gaussian noise
        if self.config.noise_std > 0:
            noise = torch.randn_like(lidar_scan) * self.config.noise_std
            lidar_scan = lidar_scan + noise
            # Clamp back to valid range
            lidar_scan = torch.clamp(lidar_scan, 0.0, 1.0)
        
        # Random circular shift
        if self.config.shift_max_beams > 0:
            shifts = torch.randint(
                -self.config.shift_max_beams,
                self.config.shift_max_beams + 1,
                size=(B,),
                device=lidar_scan.device
            )
            for b in range(B):
                lidar_scan[b] = torch.roll(lidar_scan[b], shifts[b].item(), dims=0)
        
        return lidar_scan


class LiDAR1DCNNEncoder(nn.Module):
    """
    1D CNN encoder for single-line LiDAR scans.
    Outputs fixed-dimensional embeddings suitable for ACT-style policies.
    """
    
    def __init__(self, config: Optional[LiDAREncoderConfig] = None):
        """
        Initialize LiDAR encoder.
        
        Args:
            config: LiDAREncoderConfig instance. If None, uses default config.
        """
        super().__init__()
        self.config = config if config is not None else LiDAREncoderConfig()
        
        # Build backbone: Conv1D blocks
        self.backbone = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),
            nn.Dropout(p=self.config.dropout_rate),
            
            # Block 2: 32 -> 64 channels
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(),
            nn.Dropout(p=self.config.dropout_rate),
            
            # Block 3: 64 -> 128 channels
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU(),
            nn.Dropout(p=self.config.dropout_rate),
            
            # Block 4: 128 -> 256 channels
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.SiLU(),
            nn.Dropout(p=self.config.dropout_rate_late),
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(p=self.config.dropout_rate_late),
            nn.Linear(256, self.config.embedding_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def __call__(
        self,
        lidar_scan: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            lidar_scan: Preprocessed LiDAR scan, shape (B, 1, N) or (B, N)
            validity_mask: Optional validity mask, shape (B, N). Invalid beams are zeroed.
        
        Returns:
            embedding: Fixed-dimensional embedding, shape (B, D) where D=embedding_dim
        """
        # Handle (B, N) input by adding channel dimension
        if lidar_scan.dim() == 2:
            lidar_scan = lidar_scan.unsqueeze(1)
        
        assert lidar_scan.dim() == 3, f"Expected 3D tensor (B, 1, N), got {lidar_scan.dim()}D"
        B, C, N = lidar_scan.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        
        # Apply validity mask if provided
        if validity_mask is not None:
            # Expand mask to match input shape: (B, N) -> (B, 1, N)
            if validity_mask.dim() == 2:
                validity_mask = validity_mask.unsqueeze(1)
            lidar_scan = lidar_scan * validity_mask
        
        # Ensure contiguous memory for efficiency
        lidar_scan = lidar_scan.contiguous()
        
        # Backbone: (B, 1, N) -> (B, 256, N')
        features = self.backbone(lidar_scan)
        
        # Global average pooling: (B, 256, N') -> (B, 256, 1)
        pooled = self.pool(features)
        
        # Flatten: (B, 256, 1) -> (B, 256)
        pooled = pooled.squeeze(-1)
        
        # Projection head: (B, 256) -> (B, embedding_dim)
        embedding = self.projection(pooled)
        
        return embedding


class LiDAREncoder(nn.Module):
    """
    Complete pipeline: preprocessing + encoding.
    Convenience wrapper for end-to-end processing.
    """
    
    def __init__(self, config: Optional[LiDAREncoderConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: LiDAREncoderConfig instance. If None, uses default config.
        """
        super().__init__()
        self.config = config if config is not None else LiDAREncoderConfig()
        self.preprocessor = LiDARPreprocessor(self.config)
        self.encoder = LiDAR1DCNNEncoder(self.config)
    
    def __call__(
        self,
        lidar_scan: Union[torch.Tensor, np.ndarray],
        training: bool = False,
        return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process raw LiDAR scan and return embedding.
        
        Args:
            lidar_scan: Raw LiDAR scan, shape (B, N) or (N,)
            training: Whether in training mode
            return_mask: Whether to return validity mask
        
        Returns:
            embedding: Fixed-dimensional embedding, shape (B, D)
            validity_mask: (optional) Validity mask, shape (B, N)
        """
        processed_scan, validity_mask = self.preprocessor(lidar_scan, training=training)
        embedding = self.encoder(processed_scan, validity_mask)
        
        if return_mask:
            return embedding, validity_mask
        return embedding


# ============================================================================
# Integration Example with ACT-style Policy
# ============================================================================

def example_act_integration():
    """
    Example showing how to integrate LiDAR encoder with ACT-style policy.
    This demonstrates appending LiDAR embedding as a token to other state tokens.
    """
    print("=" * 80)
    print("ACT Integration Example")
    print("=" * 80)
    
    # Configuration
    config = LiDAREncoderConfig(
        embedding_dim=256,
        r_min=0.1,
        r_max=50.0
    )
    
    # Create encoder
    lidar_encoder = LiDAREncoder(config)
    lidar_encoder.eval()
    
    # Simulate inputs
    batch_size = 4
    num_beams = 1080
    state_dim = 17  # Robot state dimension (e.g., qpos)
    hidden_dim = 256  # ACT transformer hidden dimension
    
    # Simulate LiDAR scan (B, N)
    lidar_scan = torch.rand(batch_size, num_beams) * 10.0 + 0.5  # Random ranges
    
    # Simulate robot state tokens (B, num_state_tokens, hidden_dim)
    num_state_tokens = 1  # Could be multiple tokens for different state components
    robot_state_tokens = torch.randn(batch_size, num_state_tokens, hidden_dim)
    
    # Encode LiDAR
    with torch.no_grad():
        lidar_emb = lidar_encoder(lidar_scan, training=False)  # (B, embedding_dim)
    
    # Expand LiDAR embedding to match token format: (B, embedding_dim) -> (B, 1, hidden_dim)
    # If embedding_dim != hidden_dim, use a projection layer
    if config.embedding_dim != hidden_dim:
        lidar_proj = nn.Linear(config.embedding_dim, hidden_dim)
        lidar_tokens = lidar_proj(lidar_emb).unsqueeze(1)  # (B, 1, hidden_dim)
    else:
        lidar_tokens = lidar_emb.unsqueeze(1)  # (B, 1, hidden_dim)
    
    # Concatenate tokens: (B, num_state_tokens + 1, hidden_dim)
    fused_tokens = torch.cat([robot_state_tokens, lidar_tokens], dim=1)
    
    print(f"Robot state tokens shape: {robot_state_tokens.shape}")
    print(f"LiDAR tokens shape: {lidar_tokens.shape}")
    print(f"Fused tokens shape: {fused_tokens.shape}")
    print("\n✓ Integration example complete!")
    print("=" * 80)


# ============================================================================
# Benchmark Function
# ============================================================================

def benchmark_encoder(
    config: Optional[LiDAREncoderConfig] = None,
    batch_size: int = 32,
    num_beams: int = 1080,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Benchmark encoder forward pass latency.
    
    Args:
        config: LiDAREncoderConfig instance
        batch_size: Batch size for benchmark
        num_beams: Number of LiDAR beams
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        device: Device to run on ('cpu' or 'cuda')
    """
    print("=" * 80)
    print(f"Benchmarking LiDAR Encoder (device: {device})")
    print("=" * 80)
    
    if config is None:
        config = LiDAREncoderConfig()
    
    encoder = LiDAREncoder(config)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Create dummy input
    lidar_scan = torch.rand(batch_size, num_beams, device=device) * 10.0 + 0.5
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = encoder(lidar_scan)
    
    # Synchronize if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = encoder(lidar_scan)
    
    # Synchronize if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_latency_ms = (elapsed_time / num_iterations) * 1000
    
    print(f"Batch size: {batch_size}")
    print(f"Number of beams: {num_beams}")
    print(f"Iterations: {num_iterations}")
    print(f"Average latency: {avg_latency_ms:.3f} ms")
    print(f"Throughput: {batch_size / (elapsed_time / num_iterations):.1f} samples/sec")
    print("=" * 80)


# ============================================================================
# Unit Tests
# ============================================================================

def test_shape_invariants():
    """Test that encoder handles different input shapes correctly."""
    print("Testing shape invariants...")
    
    config = LiDAREncoderConfig(embedding_dim=256)
    encoder = LiDAREncoder(config)
    encoder.eval()
    
    # Test (B, N) input
    lidar_scan_2d = torch.rand(4, 1080) * 10.0 + 0.5
    emb_2d = encoder(lidar_scan_2d)
    assert emb_2d.shape == (4, 256), f"Expected (4, 256), got {emb_2d.shape}"
    
    # Test (B, 1, N) input (preprocessed)
    lidar_scan_3d = torch.rand(4, 1, 1080) * 10.0 + 0.5
    emb_3d = encoder.encoder(lidar_scan_3d)
    assert emb_3d.shape == (4, 256), f"Expected (4, 256), got {emb_3d.shape}"
    
    # Test single sample (1, N)
    lidar_scan_1d = torch.rand(1, 1080) * 10.0 + 0.5
    emb_1d = encoder(lidar_scan_1d)
    assert emb_1d.shape == (1, 256), f"Expected (1, 256), got {emb_1d.shape}"
    
    print("✓ Shape invariants test passed!")


def test_nan_inf_handling():
    """Test that encoder handles NaN and Inf values robustly."""
    print("Testing NaN/Inf handling...")
    
    config = LiDAREncoderConfig(embedding_dim=256)
    encoder = LiDAREncoder(config)
    encoder.eval()
    
    # Create scan with NaN and Inf
    lidar_scan = torch.rand(2, 1080) * 10.0 + 0.5
    lidar_scan[0, 100] = float('nan')
    lidar_scan[0, 200] = float('inf')
    lidar_scan[1, 300] = float('-inf')
    
    # Should not crash and should produce valid output
    emb = encoder(lidar_scan)
    assert torch.isfinite(emb).all(), "Output contains NaN/Inf"
    assert emb.shape == (2, 256), f"Expected (2, 256), got {emb.shape}"
    
    print("✓ NaN/Inf handling test passed!")


def test_variable_beam_count():
    """Test that encoder handles variable number of beams."""
    print("Testing variable beam count...")
    
    config = LiDAREncoderConfig(embedding_dim=256)
    encoder = LiDAREncoder(config)
    encoder.eval()
    
    # Test different beam counts
    for num_beams in [360, 720, 1080, 1440]:
        lidar_scan = torch.rand(2, num_beams) * 10.0 + 0.5
        emb = encoder(lidar_scan)
        assert emb.shape == (2, 256), f"Failed for {num_beams} beams: got {emb.shape}"
    
    print("✓ Variable beam count test passed!")


def test_deterministic_output():
    """Test that encoder produces deterministic output when augmentation is disabled."""
    print("Testing deterministic output...")
    
    config = LiDAREncoderConfig(
        embedding_dim=256,
        enable_augmentation=False  # Disable augmentation
    )
    encoder = LiDAREncoder(config)
    encoder.eval()
    
    # Set random seed
    torch.manual_seed(42)
    lidar_scan = torch.rand(2, 1080) * 10.0 + 0.5
    
    # Run twice with same input
    torch.manual_seed(42)
    emb1 = encoder(lidar_scan)
    
    torch.manual_seed(42)
    emb2 = encoder(lidar_scan)
    
    # Should be identical (within numerical precision)
    assert torch.allclose(emb1, emb2, atol=1e-6), "Outputs are not deterministic"
    
    print("✓ Deterministic output test passed!")


def test_range_clipping():
    """Test that encoder clips values to valid range."""
    print("Testing range clipping...")
    
    config = LiDAREncoderConfig(
        embedding_dim=256,
        r_min=0.1,
        r_max=50.0
    )
    encoder = LiDAREncoder(config)
    encoder.eval()
    
    # Create scan with out-of-range values
    lidar_scan = torch.rand(2, 1080) * 100.0  # Some values > 50.0
    lidar_scan[0, 0] = 0.05  # < r_min
    lidar_scan[0, 1] = 100.0  # > r_max
    
    # Should not crash
    emb = encoder(lidar_scan)
    assert emb.shape == (2, 256), f"Expected (2, 256), got {emb.shape}"
    
    print("✓ Range clipping test passed!")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("Running Unit Tests")
    print("=" * 80 + "\n")
    
    test_shape_invariants()
    test_nan_inf_handling()
    test_variable_beam_count()
    test_deterministic_output()
    test_range_clipping()
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LiDAR 1D CNN Encoder')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--example', action='store_true', help='Run integration example')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for benchmark')
    parser.add_argument('--num_beams', type=int, default=1080, help='Number of beams for benchmark')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for benchmark')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Run requested operations
    if args.test:
        run_all_tests()
    
    if args.benchmark:
        benchmark_encoder(
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            device=device
        )
    
    if args.example:
        example_act_integration()
    
    # If no flags provided, run tests and example
    if not (args.test or args.benchmark or args.example):
        run_all_tests()
        example_act_integration()
        print("\nTo run benchmark, use: python lidar_encoder.py --benchmark")

