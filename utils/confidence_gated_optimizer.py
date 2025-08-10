"""
Confidence-Gated Optimizer (CGO) for Robust Test-Time Adaptation

This module implements a confidence-gated optimizer that conditionally applies
weight updates based on the model's prediction confidence, preventing adaptation
to highly uncertain/corrupted samples.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfidenceGatedOptimizer:
    """
    A wrapper around standard PyTorch optimizers that gates weight updates
    based on prediction confidence.
    
    The optimizer only performs weight updates when the model's prediction
    confidence (max softmax probability) exceeds a specified threshold τ.
    This prevents harmful adaptation to highly corrupted or uncertain samples.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        confidence_threshold: float = 0.7,
        confidence_metric: str = 'max_softmax',
        enable_logging: bool = False
    ):
        """
        Initialize the Confidence-Gated Optimizer.
        
        Args:
            optimizer: The underlying PyTorch optimizer (e.g., SGD, Adam)
            confidence_threshold: Threshold τ for gating updates (default: 0.7)
            confidence_metric: Method to compute confidence ('max_softmax', 'entropy')
            enable_logging: Whether to log adaptation decisions
        """
        self.optimizer = optimizer
        self.confidence_threshold = confidence_threshold
        self.confidence_metric = confidence_metric
        self.enable_logging = enable_logging
        
        # Statistics tracking
        self.total_samples = 0
        self.adapted_samples = 0
        self.skipped_samples = 0
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be in [0, 1], got {confidence_threshold}")
            
        logger.info(f"Initialized CGO with threshold={confidence_threshold}, metric={confidence_metric}")
    
    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction confidence from model logits.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            
        Returns:
            confidence: Confidence scores [batch_size]
        """
        if self.confidence_metric == 'max_softmax':
            # Use maximum softmax probability as confidence
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            
        elif self.confidence_metric == 'entropy':
            # Use negative entropy as confidence (higher entropy = lower confidence)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            # Normalize entropy to [0, 1] range (assuming max entropy is log(num_classes))
            max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
            confidence = 1.0 - (entropy / max_entropy)
            
        else:
            raise ValueError(f"Unknown confidence metric: {self.confidence_metric}")
            
        return confidence
    
    def should_adapt(self, logits: torch.Tensor) -> Tuple[bool, float]:
        """
        Determine whether to perform adaptation based on prediction confidence.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            
        Returns:
            should_adapt: Boolean indicating whether to adapt
            confidence: The computed confidence score
        """
        confidence = self.compute_confidence(logits)
        
        # For batch processing, use mean confidence
        if confidence.numel() > 1:
            mean_confidence = confidence.mean().item()
        else:
            mean_confidence = confidence.item()
            
        should_adapt = mean_confidence > self.confidence_threshold
        
        return should_adapt, mean_confidence
    
    def conditional_step(self, logits: torch.Tensor, loss: torch.Tensor) -> dict:
        """
        Conditionally perform optimizer step based on prediction confidence.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            loss: Computed loss tensor
            
        Returns:
            step_info: Dictionary with adaptation information
        """
        should_adapt, confidence = self.should_adapt(logits)
        
        step_info = {
            'confidence': confidence,
            'threshold': self.confidence_threshold,
            'adapted': should_adapt,
            'loss': loss.item() if should_adapt else 0.0
        }
        
        self.total_samples += 1
        
        if should_adapt:
            # Perform standard optimizer step
            self.optimizer.step()
            self.adapted_samples += 1
            
            if self.enable_logging:
                logger.debug(f"Adapted: confidence={confidence:.3f} > threshold={self.confidence_threshold}")
        else:
            # Skip adaptation - no weight updates
            self.skipped_samples += 1
            
            if self.enable_logging:
                logger.debug(f"Skipped: confidence={confidence:.3f} <= threshold={self.confidence_threshold}")
        
        return step_info
    
    def zero_grad(self):
        """Forward zero_grad call to underlying optimizer."""
        self.optimizer.zero_grad()
    
    def get_adaptation_stats(self) -> dict:
        """
        Get statistics about adaptation decisions.
        
        Returns:
            stats: Dictionary with adaptation statistics
        """
        adaptation_rate = self.adapted_samples / max(self.total_samples, 1)
        
        return {
            'total_samples': self.total_samples,
            'adapted_samples': self.adapted_samples,
            'skipped_samples': self.skipped_samples,
            'adaptation_rate': adaptation_rate,
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset_stats(self):
        """Reset adaptation statistics."""
        self.total_samples = 0
        self.adapted_samples = 0
        self.skipped_samples = 0
    
    def set_confidence_threshold(self, new_threshold: float):
        """
        Update the confidence threshold.
        
        Args:
            new_threshold: New threshold value in [0, 1]
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be in [0, 1], got {new_threshold}")
            
        self.confidence_threshold = new_threshold
        logger.info(f"Updated confidence threshold to {new_threshold}")
    
    def __getattr__(self, name):
        """Forward attribute access to underlying optimizer."""
        return getattr(self.optimizer, name)


class AdaptiveConfidenceGatedOptimizer(ConfidenceGatedOptimizer):
    """
    An adaptive version of CGO that dynamically adjusts the confidence threshold
    based on recent adaptation performance.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_threshold: float = 0.7,
        min_threshold: float = 0.5,
        max_threshold: float = 0.9,
        adaptation_window: int = 10,
        target_adaptation_rate: float = 0.7,
        **kwargs
    ):
        """
        Initialize Adaptive CGO.
        
        Args:
            optimizer: The underlying PyTorch optimizer
            initial_threshold: Initial confidence threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            adaptation_window: Window size for computing adaptation rate
            target_adaptation_rate: Target rate of adaptation (0.7 = 70% of samples)
        """
        super().__init__(optimizer, initial_threshold, **kwargs)
        
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_window = adaptation_window
        self.target_adaptation_rate = target_adaptation_rate
        
        # Recent adaptation history
        self.recent_adaptations = []
        
    def conditional_step(self, logits: torch.Tensor, loss: torch.Tensor) -> dict:
        """
        Perform conditional step with adaptive threshold adjustment.
        """
        step_info = super().conditional_step(logits, loss)
        
        # Track recent adaptations
        self.recent_adaptations.append(step_info['adapted'])
        if len(self.recent_adaptations) > self.adaptation_window:
            self.recent_adaptations.pop(0)
        
        # Adjust threshold if we have enough history
        if len(self.recent_adaptations) >= self.adaptation_window:
            current_rate = sum(self.recent_adaptations) / len(self.recent_adaptations)
            
            # Adjust threshold based on current vs target adaptation rate
            if current_rate < self.target_adaptation_rate - 0.1:
                # Too few adaptations - lower threshold
                new_threshold = max(self.min_threshold, self.confidence_threshold - 0.05)
            elif current_rate > self.target_adaptation_rate + 0.1:
                # Too many adaptations - raise threshold
                new_threshold = min(self.max_threshold, self.confidence_threshold + 0.05)
            else:
                new_threshold = self.confidence_threshold
            
            if new_threshold != self.confidence_threshold:
                self.set_confidence_threshold(new_threshold)
                step_info['threshold_adjusted'] = True
                step_info['new_threshold'] = new_threshold
        
        return step_info
