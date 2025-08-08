# CMGS (Confidence-Modulated Gradient Scaling) for ViTTA

## Quick Start

### Enable CMGS in main_tta.py

```python
# CMGS is automatically enabled in TTA mode
args.use_cmgs = True
args.cmgs_gamma = 0.367879  # e^(-1) - confidence threshold
args.cmgs_alpha = 1.0       # High confidence scaling
args.cmgs_beta = 0.1        # Low confidence scaling
```

### Run Multi-Epoch TTA with CMGS

```bash
# Basic usage with default parameters
python main_tta.py

# Custom CMGS parameters
python main_tta.py --use_cmgs True --cmgs_gamma 0.5 --cmgs_alpha 1.5 --cmgs_beta 0.05

# Multi-epoch adaptation
python main_tta.py --n_epoch_adapat 3 --n_gradient_steps 2
```

### Run Multiple CMGS Configurations Automatically

The `main_tta.py` script now includes a built-in loop that runs 4 different CMGS configurations sequentially:

```bash
# Run all 4 CMGS configurations in one call
python main_tta.py
```

This will automatically test:
1. **Config 1**: Alpha=1.0, Beta=0.1, Gamma=0.367879, Epochs=4
2. **Config 2**: Alpha=2.0, Beta=0.1, Gamma=0.367879, Epochs=8
3. **Config 3**: Alpha=2.0, Beta=0.1, Gamma=0.2, Epochs=4
4. **Config 4**: Alpha=2.0, Beta=0.1, Gamma=0.2, Epochs=8

Results are saved in separate directories for each configuration.

## What is CMGS?

**Confidence-Modulated Gradient Scaling (CMGS)** is a lightweight technique for robust Test-Time Adaptation (TTA) under severe and unpredictable corruptions.

### Key Features

1. **Per-Sample Confidence Estimation**: Computes confidence scores based on softmax probabilities
2. **Adaptive Gradient Scaling**: Amplifies gradients from high-confidence samples, dampens from low-confidence ones
3. **Multi-Epoch Support**: Compatible with multi-epoch TTA adaptation
4. **Lightweight**: No additional model parameters, minimal computational overhead
5. **Automated Testing**: Built-in loop for testing multiple parameter configurations

### How It Works

1. **Confidence Computation**: For each sample $x_i$, compute confidence $c_i = \max(p_i)$
2. **Gradient Modulation**: 
   - High-confidence samples ($c_i > \gamma$): Amplified gradients (×α)
   - Low-confidence samples ($c_i < \gamma$): Dampened gradients (×β)
3. **Loss Function**: $L_{\text{CMGS}} = c_i \cdot \log\left(\frac{e^\gamma}{c_i}\right)$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_cmgs` | `False` | Enable CMGS |
| `cmgs_gamma` | `0.367879` | Confidence threshold (e^(-1)) |
| `cmgs_alpha` | `1.0` | High-confidence scaling factor |
| `cmgs_beta` | `0.1` | Low-confidence scaling factor |
| `n_epoch_adapat` | `2` | Number of adaptation epochs |

## Usage Examples

### Conservative CMGS
```python
args.cmgs_gamma = 0.7    # Higher threshold
args.cmgs_alpha = 1.5    # More aggressive high-conf
args.cmgs_beta = 0.05    # More aggressive low-conf
```

### Aggressive CMGS
```python
args.cmgs_gamma = 0.2    # Lower threshold
args.cmgs_alpha = 2.0    # Very aggressive high-conf
args.cmgs_beta = 0.01    # Very aggressive low-conf
```

## Testing

### Run Test Suite
```bash
python test_cmgs.py
```

### Run Examples
```bash
python example_cmgs_usage.py
```

### Test Configuration Loop
```bash
python test_cmgs_loop.py
```

## Files Modified

1. **main_tta.py**: Added CMGS parameters, configuration loop, and multi-config testing
2. **utils/opts.py**: Added CMGS command-line arguments
3. **corpus/test_time_adaptation.py**: 
   - Added `compute_cmgs_loss()` function
   - Integrated CMGS into TTA training loop
4. **test_cmgs.py**: Comprehensive test suite
5. **test_cmgs_loop.py**: Test script for configuration loop
6. **example_cmgs_usage.py**: Usage examples
7. **docs/CMGS_IMPLEMENTATION.md**: Detailed documentation

## Performance Benefits

- **Robust Adaptation**: Prevents overfitting to noisy/corrupted samples
- **Faster Convergence**: Amplified gradients accelerate learning from reliable inputs
- **Better Accuracy**: Improved performance under random per-sample corruptions
- **Multi-Epoch Support**: Enables longer adaptation periods for better results
- **Automated Testing**: Built-in loop for systematic parameter exploration

## Theoretical Foundation

CMGS is based on information-theoretic principles:
- **Confidence as Information**: High confidence indicates reliable information
- **Gradient Modulation**: Amplify gradients from reliable sources, dampen from unreliable ones
- **Robust Adaptation**: Prevents catastrophic forgetting while enabling rapid adaptation

## Citation

If you use CMGS in your research, please cite:

```bibtex
@article{cmgs2024,
  title={Confidence-Modulated Gradient Scaling for Robust Test-Time Adaptation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact

For questions or issues with the CMGS implementation, please refer to the main ViTTA repository or create an issue in the project repository. 