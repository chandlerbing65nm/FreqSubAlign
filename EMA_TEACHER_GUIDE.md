# EMA Teacher Self-Distillation Guide for ViTTA

This guide documents the implementation of **Inverted Teacher-Student Self-Distillation** in ViTTA, which provides label-free test-time adaptation using an Exponential Moving Average (EMA) teacher model.

## Overview

Traditional ViTTA requires test labels for cross-entropy loss, which is problematic for true test-time adaptation. This implementation introduces a **label-free distillation loss** using an EMA teacher that:

- **Generates soft targets** from the teacher's predictions
- **Focuses on hard samples** using confidence-inverted weighting (aligned with CIMO philosophy)
- **Requires no test labels** - fully unsupervised TTA
- **Integrates seamlessly** with existing ViTTA pipeline

## Technical Design

### Core Components

1. **EMA Teacher (`utils/ema_teacher.py`)**
   - Maintains a slowly-updated copy of the student model
   - Provides stable soft targets for distillation
   - Supports adaptive temperature based on confidence

2. **Confidence-Inverted Weighting**
   - Weights distillation loss inversely with confidence
   - Focuses learning on uncertain/hard samples
   - Aligns with CIMO's "learn from mistakes" philosophy

3. **Integration Points**
   - Added to `corpus/test_time_adaptation.py` TTA pipeline
   - Configurable via command-line arguments
   - Preserves backward compatibility

## Usage

### Basic Usage

```bash
# Enable EMA teacher with default settings
python main_tta.py --use_ema_teacher --tta --use_cimo

# With custom parameters
python main_tta.py --use_ema_teacher \
                   --ema_momentum 0.995 \
                   --ema_temperature 3.0 \
                   --lambda_distill 0.5 \
                   --tta --use_cimo
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_ema_teacher` | False | Enable EMA teacher distillation |
| `--ema_momentum` | 0.999 | EMA momentum (higher = slower teacher updates) |
| `--ema_temperature` | 4.0 | Base temperature for soft targets |
| `--ema_adaptive_temp` | True | Adapt temperature based on confidence |
| `--ema_min_temp` | 1.0 | Minimum temperature for adaptive scaling |
| `--ema_max_temp` | 8.0 | Maximum temperature for adaptive scaling |
| `--ema_temp_alpha` | 2.0 | Temperature adaptation scaling factor |
| `--lambda_distill` | 1.0 | Weight for distillation loss |

### Recommended Settings

#### Conservative (Stable)
```bash
--ema_momentum 0.999 --ema_temperature 2.0 --lambda_distill 0.5
```

#### Aggressive (Adaptive)
```bash
--ema_momentum 0.995 --ema_temperature 4.0 --lambda_distill 1.0 --ema_adaptive_temp
```

#### Balanced
```bash
--ema_momentum 0.997 --ema_temperature 3.0 --lambda_distill 0.8
```

## Integration with Optimizers

### With CIMO (Confidence-Inverted Meta-Optimizer)
```bash
python main_tta.py --tta \
                   --use_cimo \
                   --cimo_confidence_threshold 0.3 \
                   --use_ema_teacher \
                   --lambda_distill 1.0
```

### With CGO (Confidence-Gated Optimizer)
```bash
python main_tta.py --tta \
                   --use_cgo \
                   --cgo_confidence_threshold 0.7 \
                   --use_ema_teacher \
                   --lambda_distill 0.8
```

### Multi-Epoch TTA
```bash
python main_tta.py --tta \
                   --n_epoch_adapat 4 \
                   --use_cimo \
                   --use_ema_teacher \
                   --ema_momentum 0.995
```

## How It Works

### 1. Teacher Initialization
```python
# During TTA setup
ema_teacher = EMATeacher(
    model=model,
    momentum=args.ema_momentum,
    temperature=args.ema_temperature,
    adaptive_temperature=args.ema_adaptive_temp,
    device=device
)
```

### 2. Distillation Loss Computation
```python
# In adaptation loop
if use_ema_teacher:
    # Get teacher predictions
    teacher_logits = ema_teacher.forward(input, confidence)
    
    # Compute distillation loss
    loss_distill = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )
    
    # Apply confidence-inverted weighting
    inverted_weight = (max_temp - alpha * (1 - confidence))
    loss_distill = loss_distill * inverted_weight.mean()
```

### 3. Teacher Update
```python
# After optimizer step
ema_teacher.update(model)
```

## Result Suffix

The implementation automatically includes EMA teacher parameters in the result suffix:

```
Results saved with suffix: 
"celoss=False_adaptepoch=4_cimo_tau0.3_power2.0_ema0.999_temp4.0_adapttemp_distill1.0"
```

## Testing

Run the comprehensive test suite:

```bash
python test_ema_teacher.py
```

This tests:
- Basic EMA teacher functionality
- Confidence-inverted weighting
- Distillation loss computation
- Integration with ViTTA pipeline

## Performance Tips

### For Fish Feeding Classification
1. **Start conservative**: Use `ema_momentum=0.999` and `lambda_distill=0.5`
2. **Adapt temperature**: Enable `ema_adaptive_temp=True` for varying corruption levels
3. **Multi-epoch**: Use `n_epoch_adapat=4-6` for better adaptation

### For Cross-Domain Adaptation
1. **Higher momentum**: Use `ema_momentum=0.9995` when using pretrained statistics
2. **Lower temperature**: Use `ema_temperature=2.0-3.0` for sharper targets
3. **Moderate weight**: Use `lambda_distill=0.3-0.7` to avoid overfitting

## Troubleshooting

### Common Issues

1. **Zero distillation loss**: Check if `use_ema_teacher=True` is set
2. **NaN in loss**: Reduce `lambda_distill` or increase `ema_temperature`
3. **Poor adaptation**: Try increasing `ema_momentum` or decreasing `lambda_distill`

### Debug Mode
```bash
python main_tta.py --verbose --use_ema_teacher ...
```

## Architecture Compatibility

- **TANet**: Fully supported
- **Video Swin Transformer**: Fully supported
- **Custom models**: Requires standard forward() method

## Future Extensions

- **Multiple teachers**: Ensemble of EMA teachers
- **Prototypical targets**: Cluster-based soft targets
- **Dynamic weighting**: Adaptive `lambda_distill` based on corruption level

## Citation

If you use this EMA teacher self-distillation in your research:

```bibtex
@misc{vitta_ema_teacher,
  title={ViTTA: Video Test-Time Adaptation with EMA Teacher Self-Distillation},
  author={Your Name},
  year={2024},
  howpublished={ViTTA Extension Implementation}
}
```
