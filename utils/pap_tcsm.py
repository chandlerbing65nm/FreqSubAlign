import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def _iter_bn_layers(model):
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            yield name, layer


def run_pap_warmup(model: torch.nn.Module,
                   x_probe: torch.Tensor,
                   args: Optional[object] = None,
                   logger: Optional[object] = None) -> int:
    """
    Pre-Adaptation Probing (PAP) with optional Stochastic BatchNorm Momentum (SBM) and
    Temporally-Correlated Stochastic Momentum (TCSM) used to warm up AMP/loss-scale and
    calibrate BN running stats before adaptation.

    Behavior mirrors the inlined logic previously in tta_standard.

    - Runs forward-only passes (no grad)
    - Optionally modulates BN.momentum per layer between [mom_min, mom_max]
    - Supports IID stochastic gating or TCSM AR(1)-like correlated gating

    Returns number of successful probe passes.
    """
    if args is None or not getattr(args, 'probe_ffp_enable', False):
        return 0

    prev_mode = model.training
    # ensure BN running stats can update
    model.train(True)

    # Only apply BN momentum modulation for TANet (BN-based). For others, forward-only warmup still helps AMP.
    use_cg_bnmm = bool(getattr(args, 'probe_cg_bnmm_enable', False)) and getattr(args, 'arch', '') == 'tanet'

    mom_min = 0.01
    mom_max = 0.10

    # TCSM controls
    tcsm_enable = bool(getattr(args, 'probe_sbm_tcsm_enable', False))
    tcsm_rho = float(getattr(args, 'probe_sbm_rho', 0.8))
    tcsm_prev_gate = float(getattr(args, 'probe_sbm_init', 0.5))

    # Track original BN momentum to restore after probes
    orig_momentum = {}
    if use_cg_bnmm:
        for layer_name, layer in _iter_bn_layers(model):
            orig_momentum[layer_name] = layer.momentum

    probes_run = 0
    backoff_used = 0

    # Initialize per-layer momentum for first pass
    next_momentum = {}
    if use_cg_bnmm:
        init_m = (mom_min + mom_max) * 0.5
        for layer_name, layer in _iter_bn_layers(model):
            next_momentum[layer_name] = init_m

    steps = int(getattr(args, 'probe_ffp_steps', 1))
    use_amp = bool(getattr(args, 'probe_amp', True)) and torch.cuda.is_available()

    for k in range(steps):
        try:
            with torch.no_grad():
                # Apply per-layer momentum before this probe forward
                if use_cg_bnmm:
                    for layer_name, layer in _iter_bn_layers(model):
                        if layer_name in next_momentum:
                            layer.momentum = float(next_momentum[layer_name])

                if use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        _ = model(x_probe)
                else:
                    _ = model(x_probe)

                # Compute SBM gate in [0,1]
                if tcsm_enable:
                    # Temporally-Correlated: convex-combination AR(1)-like update
                    u = float(torch.rand(1).item())
                    conf = float(tcsm_rho * tcsm_prev_gate + (1.0 - tcsm_rho) * u)
                    conf = min(max(conf, 0.0), 1.0)
                    tcsm_prev_gate = conf
                else:
                    # IID random gate — here deterministically trust=1 for stability; adjust if desired
                    conf = 1.0

                # Update per-layer momentum for the next pass
                if use_cg_bnmm:
                    updated_momentum = {}
                    trust = float(min(max(conf, 0.0), 1.0))
                    for layer_name, layer in _iter_bn_layers(model):
                        m_val = mom_min + trust * (mom_max - mom_min)
                        updated_momentum[layer_name] = m_val
                    next_momentum = updated_momentum

                if logger is not None:
                    if use_cg_bnmm:
                        avg_m = np.mean(list(next_momentum.values())) if len(next_momentum) > 0 else float('nan')
                        mode = 'TCSM' if tcsm_enable else 'IID'
                        logger.debug(f"[PAP][Stochastic BatchNorm Momentum:{mode}] probe {k+1}: gate={conf:.4f}, avg_momentum={avg_m:.4f}, amp={use_amp}")
                    else:
                        logger.debug(f"[PAP] probe {k+1}: gate={conf:.4f}, amp={use_amp}")

                probes_run += 1
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                backoff_used += 1
                if logger is not None:
                    logger.warning(f"[PAP] OOM during probe pass {k+1}. Backoff {backoff_used}/{getattr(args, 'probe_max_backoff', 1)}; skipping remaining probes.")
                if backoff_used >= int(getattr(args, 'probe_max_backoff', 1)):
                    break
            else:
                if logger is not None:
                    logger.warning(f"[PAP] Probe error: {e}")
                break
        except Exception as e:
            if logger is not None:
                logger.warning(f"[PAP] Probe exception: {e}")
            break

    # Restore BN momentum
    if use_cg_bnmm:
        for layer_name, layer in _iter_bn_layers(model):
            if layer_name in orig_momentum:
                layer.momentum = orig_momentum[layer_name]

    if logger is not None:
        logger.info(f"[PAP] Completed {probes_run} forward-only probe pass(es); model warmed up.")

    model.train(prev_mode)
    return probes_run
