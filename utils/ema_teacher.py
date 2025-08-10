"""
EMA Teacher for Inverted Teacher-Student Self-Distillation in ViTTA
Implements Exponential Moving Average teacher for label-free distillation
with confidence-inverted weighting aligned with CIMO philosophy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EMATeacher:
    """
    Exponential Moving Average teacher for self-distillation in TTA.
    
    Provides softened predictions that serve as targets for distillation loss,
    with confidence-inverted weighting to focus on hard samples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.999,
        temperature: float = 4.0,
        adaptive_temperature: bool = True,
        min_temperature: float = 1.0,
        max_temperature: float = 8.0,
        temperature_alpha: float = 2.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EMA Teacher.
        
        Args:
            model: Student model to create teacher from
            momentum: EMA momentum (higher = slower teacher updates)
            temperature: Base temperature for soft targets
            adaptive_temperature: Whether to adapt temperature based on confidence
            min_temperature: Minimum temperature when adapting
            max_temperature: Maximum temperature when adapting
            temperature_alpha: Scaling factor for temperature adaptation
            device: Device to place teacher on
        """
        self.momentum = momentum
        self.temperature = temperature
        self.adaptive_temperature = adaptive_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature_alpha = temperature_alpha
        self.device = device or next(model.parameters()).device
        
        # Create teacher model (copy of student)
        self.teacher = self._create_teacher(model)
        self.teacher.eval()
        
        logger.info(f"EMA Teacher initialized with momentum={momentum}, "
                   f"temperature={temperature}, adaptive={adaptive_temperature}")
    
    def _create_teacher(self, model: nn.Module) -> nn.Module:
        """Create teacher model as deep copy of student."""
        try:
            # Try to create new instance with same type
            teacher = type(model)(**model.__dict__.get('_init_args', {}))
        except (TypeError, ValueError):
            # Fallback: create deep copy of the model
            import copy
            teacher = copy.deepcopy(model)
        
        teacher.load_state_dict(model.state_dict())
        teacher.to(self.device)
        return teacher
    
    def update(self, model: nn.Module):
        """Update teacher parameters with EMA of student parameters."""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), model.parameters()
            ):
                teacher_param.data.mul_(self.momentum).add_(
                    student_param.data, alpha=1 - self.momentum
                )
    
    def get_temperature(self, confidence: torch.Tensor) -> torch.Tensor:
        """Get adaptive temperature based on confidence."""
        if not self.adaptive_temperature:
            return torch.tensor(self.temperature, device=confidence.device)
        
        # Temperature decreases as confidence increases (inverted relationship)
        temp = self.max_temperature - self.temperature_alpha * (1 - confidence)
        return torch.clamp(temp, self.min_temperature, self.max_temperature)
    
    def forward(
        self,
        x: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through teacher to get soft targets.
        
        Args:
            x: Input tensor
            confidence: Confidence tensor for adaptive temperature
            
        Returns:
            Soft targets (logits or probabilities)
        """
        with torch.no_grad():
            if hasattr(self.teacher, 'forward_features'):
                # Handle Video Swin Transformer
                logits = self.teacher(x)
            else:
                # Handle TANet or other models
                logits = self.teacher(x)
            
            # Return raw logits; temperature scaling is handled at loss site
            return logits
    
    def get_soft_targets(
        self,
        x: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get soft probability targets from teacher."""
        logits = self.forward(x, confidence)
        return F.softmax(logits, dim=-1)
    
    def state_dict(self) -> Dict[str, Any]:
        """Save teacher state."""
        return {
            'teacher': self.teacher.state_dict(),
            'momentum': self.momentum,
            'temperature': self.temperature,
            'adaptive_temperature': self.adaptive_temperature,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature,
            'temperature_alpha': self.temperature_alpha
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load teacher state."""
        self.teacher.load_state_dict(state_dict['teacher'])
        self.momentum = state_dict['momentum']
        self.temperature = state_dict['temperature']
        self.adaptive_temperature = state_dict['adaptive_temperature']
        self.min_temperature = state_dict['min_temperature']
        self.max_temperature = state_dict['max_temperature']
        self.temperature_alpha = state_dict['temperature_alpha']
