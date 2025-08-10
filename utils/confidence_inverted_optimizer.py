"""
Confidence-Inverted Meta-Optimizer (CIMO) for ViTTA
Inverted approach: learns from hard/unconfident samples instead of easy/confident ones
This can lead to better robustness by focusing on challenging cases.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceInvertedOptimizer:
    """
    Inverted optimizer that uses low confidence to drive stronger learning
    and high confidence to apply gentle updates.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        confidence_threshold: float = 0.3,
        confidence_metric: str = 'max_softmax',
        min_lr_scale: float = 0.1,
        max_lr_scale: float = 3.0,
        confidence_power: float = 2.0,
        enable_momentum_correction: bool = True,
        adaptation_history_size: int = 100,
        enable_logging: bool = False
    ):
        """
        Initialize Confidence-Inverted Optimizer.
        
        Args:
            optimizer: Base PyTorch optimizer
            confidence_threshold: Reference confidence threshold (inverted logic)
            confidence_metric: Metric for confidence calculation ('max_softmax', 'entropy')
            min_lr_scale: Minimum learning rate scaling factor
            max_lr_scale: Maximum learning rate scaling factor (higher for low confidence)
            confidence_power: Power for inverted confidence scaling
            enable_momentum_correction: Whether to adjust momentum based on confidence
            adaptation_history_size: Size of adaptation history buffer
            enable_logging: Whether to enable detailed logging
        """
        self.optimizer = optimizer
        self.confidence_threshold = confidence_threshold
        self.confidence_metric = confidence_metric
        self.min_lr_scale = min_lr_scale
        self.max_lr_scale = max_lr_scale
        self.confidence_power = confidence_power
        self.enable_momentum_correction = enable_momentum_correction
        self.enable_logging = enable_logging
        
        # Statistics tracking
        self.total_samples = 0
        self.adaptation_history = deque(maxlen=adaptation_history_size)
        self.confidence_history = deque(maxlen=adaptation_history_size)
        
        # Store original learning rates
        self.original_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute prediction confidence from logits."""
        if self.confidence_metric == 'max_softmax':
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
        elif self.confidence_metric == 'entropy':
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
            confidence = 1.0 - (entropy / max_entropy)
        else:
            raise ValueError(f"Unknown confidence metric: {self.confidence_metric}")
        
        return confidence
    
    def compute_lr_scale(self, confidence: torch.Tensor) -> float:
        """
        Compute INVERTED learning rate scaling based on confidence.
        
        Args:
            confidence: Confidence tensor [batch_size]
            
        Returns:
            lr_scale: Inverted learning rate scaling factor
        """
        # Use mean confidence for batch processing
        if confidence.numel() > 1:
            mean_confidence = confidence.mean().item()
        else:
            mean_confidence = confidence.item()
        
        # INVERTED scaling: lower confidence = stronger updates
        # Use (1 - confidence) to invert the relationship
        inverted_conf = 1.0 - mean_confidence
        normalized_inverted = inverted_conf ** self.confidence_power
        
        # Map to learning rate scale range
        lr_scale = self.min_lr_scale + (self.max_lr_scale - self.min_lr_scale) * normalized_inverted
        
        return lr_scale
    
    def compute_gradient_weight(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Compute INVERTED per-sample gradient weighting based on confidence.
        
        Args:
            confidence: Confidence tensor [batch_size]
            
        Returns:
            weights: Inverted per-sample weights [batch_size]
        """
        # INVERTED weighting: lower confidence = higher weight
        inverted_conf = 1.0 - confidence
        weights = inverted_conf ** self.confidence_power
        weights = torch.clamp(weights, min=0.1, max=3.0)  # Prevent extreme weights
        
        return weights
    
    def conditional_step(self, logits: torch.Tensor, loss: torch.Tensor) -> Dict:
        """
        Perform confidence-inverted optimizer step.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            loss: Computed loss tensor
            
        Returns:
            step_info: Dictionary with adaptation information
        """
        confidence = self.compute_confidence(logits)
        lr_scale = self.compute_lr_scale(confidence)
        gradient_weights = self.compute_gradient_weight(confidence)
        
        # Apply gradient weighting by scaling the loss
        weighted_loss = loss * gradient_weights.mean()
        
        # Scale learning rates based on INVERTED confidence
        for i, (group, original_lr) in enumerate(zip(self.optimizer.param_groups, self.original_lrs)):
            group['lr'] = original_lr * lr_scale
        
        # Perform optimizer step with scaled learning rate
        self.optimizer.step()
        
        # Restore original learning rates
        for i, (group, original_lr) in enumerate(zip(self.optimizer.param_groups, self.original_lrs)):
            group['lr'] = original_lr
        
        # Record statistics
        mean_confidence = confidence.mean().item()
        self.total_samples += 1
        self.adaptation_history.append(True)
        self.confidence_history.append(mean_confidence)
        
        step_info = {
            'confidence': mean_confidence,
            'lr_scale': lr_scale,
            'gradient_weight': gradient_weights.mean().item(),
            'threshold': self.confidence_threshold,
            'adapted': True,
            'loss': loss.item(),
            'inverted_logic': True  # Flag to indicate inverted approach
        }
        
        if self.enable_logging:
            logger.debug(f"CIMO Step - Confidence: {mean_confidence:.3f}, "
                        f"LR Scale: {lr_scale:.3f} (INVERTED), "
                        f"Gradient Weight: {gradient_weights.mean().item():.3f}")
        
        return step_info
    
    def get_adaptation_stats(self) -> Dict:
        """Get adaptation statistics."""
        if len(self.confidence_history) == 0:
            return {
                'total_samples': self.total_samples,
                'mean_confidence': 0.0,
                'mean_lr_scale': 0.0,
                'adaptation_rate': 1.0,
                'inverted_logic': True
            }
        
        return {
            'total_samples': self.total_samples,
            'mean_confidence': np.mean(self.confidence_history),
            'mean_lr_scale': np.mean([self.compute_lr_scale(torch.tensor([c])) 
                                     for c in self.confidence_history]),
            'adaptation_rate': 1.0,
            'inverted_logic': True
        }
    
    def reset_stats(self):
        """Reset adaptation statistics."""
        self.total_samples = 0
        self.adaptation_history.clear()
        self.confidence_history.clear()
    
    # PyTorch optimizer compatibility methods
    def zero_grad(self):
        """Zero gradients of the wrapped optimizer."""
        return self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dict of the wrapped optimizer."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into the wrapped optimizer."""
        return self.optimizer.load_state_dict(state_dict)
    
    def __getattr__(self, name):
        """Delegate other method calls to the wrapped optimizer."""
        return getattr(self.optimizer, name)


class AdaptiveConfidenceInvertedOptimizer(ConfidenceInvertedOptimizer):
    """
    Adaptive version of CIMO that adjusts confidence threshold based on adaptation history.
    """
    
    def __init__(self, *args, target_uncertainty: float = 0.7, adaptation_window: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_uncertainty = target_uncertainty
        self.adaptation_window = adaptation_window
        
    def compute_lr_scale(self, confidence: torch.Tensor) -> float:
        """
        Adaptive learning rate scaling that considers recent uncertainty levels.
        """
        mean_confidence = confidence.mean().item()
        
        # Adjust threshold based on recent confidence history
        if len(self.confidence_history) >= self.adaptation_window:
            recent_mean = np.mean(list(self.confidence_history)[-self.adaptation_window:])
            recent_uncertainty = 1.0 - recent_mean
            
            # Adjust target to maintain desired uncertainty level
            threshold_adjustment = self.target_uncertainty - recent_uncertainty
            
            # Smooth adjustment
            adjusted_threshold = max(0.1, min(0.9, 
                self.confidence_threshold + 0.1 * threshold_adjustment))
            
            # Use adjusted threshold for scaling
            inverted_conf = 1.0 - mean_confidence
        else:
            inverted_conf = 1.0 - mean_confidence
        
        normalized_inverted = inverted_conf ** self.confidence_power
        lr_scale = self.min_lr_scale + (self.max_lr_scale - self.min_lr_scale) * normalized_inverted
        
        return lr_scale
