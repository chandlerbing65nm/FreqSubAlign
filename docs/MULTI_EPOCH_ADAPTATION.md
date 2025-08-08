# Multi-Epoch Test-Time Adaptation (TTA) in ViTTA

This document describes the multi-epoch TTA functionality implemented in ViTTA, which allows for multiple adaptation epochs during test-time adaptation to improve model performance on corrupted video data.

## Overview

The original ViTTA implementation performed test-time adaptation with a single epoch per sample. This multi-epoch extension allows the model to adapt for multiple epochs on each test sample, potentially leading to better adaptation and improved performance on corrupted video data.

## Key Features

- **Backward Compatibility**: Default behavior remains unchanged (1 epoch) for existing code
- **Configurable Epochs**: Support for any number of adaptation epochs via `--n_epoch_adapat` parameter
- **Memory Efficient**: Proper memory management across multiple epochs
- **Mode Support**: Works with both `tta_standard` and `tta_online` modes
- **Architecture Agnostic**: Compatible with both TANet and Video Swin Transformer

## Usage

### Command Line Usage

```bash
# Single epoch (default, backward compatible)
python main_tta.py --n_epoch_adapat 1

# Multi-epoch adaptation
python main_tta.py --n_epoch_adapat 5

# With other TTA parameters
python main_tta.py --n_epoch_adapat 3 --lr 1e-5 --lambda_feature_reg 1.0
```

### Programmatic Usage

```python
from utils.opts import get_opts
from corpus.main_eval import eval

# Get arguments
args = get_opts()

# Set multi-epoch parameters
args.n_epoch_adapat = 5  # Number of adaptation epochs
args.tta = True
args.arch = 'tanet'  # or 'videoswintransformer'

# Run evaluation with multi-epoch TTA
results, _ = eval(args=args)
```

## Configuration Parameters

### Core Parameters

- `--n_epoch_adapat`: Number of adaptation epochs (default: 1)
  - Type: `int`
  - Range: ≥ 1
  - Description: Controls how many epochs the model adapts on each test sample

### TTA Mode Compatibility

#### TTA Standard Mode (`tta_standard`)
- **Behavior**: Model is reinitialized for each sample, then adapted for `n_epoch_adapat` epochs
- **Requirements**: `momentum_mvg = 1.0` (no accumulation across samples)
- **Use Case**: When you want fresh adaptation for each sample

#### TTA Online Mode (`tta_online`)
- **Behavior**: Model state persists across samples, adapted for `n_epoch_adapat` epochs per sample
- **Requirements**: `momentum_mvg ≠ 1.0` (accumulation across samples), `n_gradient_steps = 1`
- **Use Case**: When you want continuous adaptation across the test set

## Implementation Details

### Architecture Changes

The multi-epoch functionality is implemented in `corpus/test_time_adaptation.py` with the following key changes:

1. **Epoch Loop Addition**: Added an outer epoch loop around the existing adaptation logic
2. **Input Preservation**: Original input/target tensors are preserved for multiple epochs
3. **Model State Management**: Proper handling of model and optimizer state across epochs
4. **Hook Management**: Statistics hooks are properly managed across epochs

### Code Structure

```python
for batch_id, (input, target) in enumerate(tta_loader):
    # Store original input/target for multiple epochs
    original_input, original_target = input.clone(), target.clone()
    
    # Multi-epoch training loop
    for epoch_id in range(args.n_epoch_adapat):
        # Use original input/target for each epoch
        input, target = original_input.clone(), original_target.clone()
        
        # Setup model/optimizer (if needed)
        if setup_model_optimizer:
            # Initialize model, hooks, optimizer
            
        # Adaptation training
        for step_id in range(n_gradient_steps):
            # Forward pass, loss computation, backward pass
            
        # Inference and evaluation
        # Hook cleanup
```

### Memory Management

- **Input Cloning**: Original inputs are cloned to prevent modification across epochs
- **Hook Cleanup**: Statistics hooks are properly closed after each epoch
- **GPU Memory**: Efficient memory usage with proper cleanup between epochs

## Performance Considerations

### Benefits of Multi-Epoch Adaptation

1. **Better Convergence**: More epochs allow the model to better adapt to test-time distribution shifts
2. **Improved Statistics**: Multiple epochs provide more stable feature statistics
3. **Enhanced Performance**: Empirically shown to improve accuracy on corrupted data

### Computational Overhead

- **Time Complexity**: Linear increase with number of epochs (O(n) where n = `n_epoch_adapat`)
- **Memory Usage**: Minimal additional memory overhead due to input cloning
- **GPU Utilization**: Better GPU utilization with longer adaptation per sample

### Recommended Settings

| Scenario | Recommended Epochs | Rationale |
|----------|-------------------|-----------|
| Light Corruption | 1-2 | Minimal adaptation needed |
| Moderate Corruption | 3-5 | Balanced adaptation vs. speed |
| Heavy Corruption | 5-10 | More adaptation for severe shifts |
| Research/Ablation | 1, 3, 5, 10 | Common values for comparison |

## Testing

A comprehensive test suite is provided in `test_multi_epoch_tta.py`:

```bash
# Run the test suite
python test_multi_epoch_tta.py
```

### Test Coverage

- Argument parsing validation
- Backward compatibility testing
- Multi-epoch configuration testing
- TTA mode compatibility
- Epoch loop structure validation
- Statistics handling verification
- Memory management testing

## Examples

### Example 1: Basic Multi-Epoch TTA

```python
# Configure for 3-epoch adaptation
args.n_epoch_adapat = 3
args.arch = 'videoswintransformer'
args.dataset = 'ucf101'
args.tta = True

# Run evaluation
results, _ = eval(args=args)
```

### Example 2: Fish Feeding Domain Adaptation

```python
# Multi-epoch adaptation for fish feeding videos
args.n_epoch_adapat = 5  # More epochs for domain shift
args.arch = 'tanet'
args.dataset = 'uffia'  # Fish feeding dataset
args.lr = 1e-5  # Lower learning rate for stability
args.lambda_feature_reg = 1.0
```

### Example 3: Ablation Study

```python
# Test different epoch counts
epoch_counts = [1, 2, 3, 5, 10]
results = {}

for epochs in epoch_counts:
    args.n_epoch_adapat = epochs
    result, _ = eval(args=args)
    results[epochs] = result
    print(f"Epochs: {epochs}, Accuracy: {result[0]:.3f}")
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or number of epochs
2. **Slow Performance**: Consider reducing epochs for faster evaluation
3. **Convergence Issues**: Adjust learning rate or regularization parameters

### Debug Mode

Enable verbose logging to monitor multi-epoch adaptation:

```python
args.verbose = True
args.n_epoch_adapat = 3
```

This will print detailed information about each epoch's progress.

## Future Enhancements

### Potential Improvements

1. **Adaptive Epochs**: Automatically determine optimal number of epochs per sample
2. **Early Stopping**: Stop adaptation when convergence is reached
3. **Epoch Scheduling**: Different learning rates for different epochs
4. **Statistics Accumulation**: Better handling of statistics across epochs

### Research Directions

1. **Optimal Epoch Count**: Systematic study of epoch count vs. performance
2. **Domain-Specific Tuning**: Epoch count optimization for specific domains
3. **Computational Efficiency**: Methods to reduce computational overhead

## References

- Original ViTTA paper: "Video Test-Time Adaptation for Action Recognition"
- Implementation based on the official ViTTA codebase
- Multi-epoch extension for improved test-time adaptation

## Contributing

When contributing to the multi-epoch TTA functionality:

1. Ensure backward compatibility (default `n_epoch_adapat = 1`)
2. Add appropriate tests to `test_multi_epoch_tta.py`
3. Update documentation as needed
4. Consider memory and computational efficiency

## License

This multi-epoch TTA implementation follows the same license as the original ViTTA codebase.
