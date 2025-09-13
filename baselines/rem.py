import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# Reuse TENT utilities for entropy and BN-only config
from .tent import softmax_entropy
import baselines.tent as tent


def _ensure_ncthw(x: torch.Tensor, arch: Optional[str]) -> Tuple[torch.Tensor, Optional[Tuple[int, ...]]]:
    """
    Ensure input is in N, C, T, H, W format for masking when needed.
    Returns the possibly-permuted tensor and a tuple describing how to invert for TANet.

    Notes:
    - TANet path in validate() provides input as N, T, C, H, W; we permute to N, C, T, H, W for masking.
    - VideoSwin path provides input as N, V, C, T, H, W; we leave as-is and handle 6D masking downstream.
    - If already N, C, T, H, W, we return as-is.
    """
    # TANet adaptation path uses N, T, C, H, W
    if arch == 'tanet' and x.ndim == 5 and x.shape[2] in (3,):
        # Likely already N, T, C, H, W
        return x.permute(0, 2, 1, 3, 4).contiguous(), (0, 2, 1, 3, 4)  # to N,C,T,H,W and remember inverse
    # VideoSwin uses N, V, C, T, H, W; leave unchanged
    return x, None


def _invert_to_tanet(x_ncthw: torch.Tensor, inv_perm: Optional[Tuple[int, ...]]) -> torch.Tensor:
    if inv_perm is None:
        return x_ncthw
    # Inverse permutation to N, T, C, H, W
    return x_ncthw.permute(inv_perm).contiguous()


def _apply_spatial_mask(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Apply a random spatial mask shared across channels and time.

    Supports:
    - 5D: [N, C, T, H, W] -> mask shape [N, 1, 1, H, W]
    - 6D: [N, V, C, T, H, W] -> mask shape [N, 1, 1, 1, H, W] (shared across views)
    """
    if ratio <= 0:
        return x
    keep_prob = 1.0 - ratio
    if x.ndim == 5:
        N, C, T, H, W = x.shape
        mask = torch.bernoulli(
            torch.full((N, 1, 1, H, W), keep_prob, device=x.device, dtype=x.dtype)
        )
        return x * mask
    elif x.ndim == 6:
        N, V, C, T, H, W = x.shape
        # Share mask across views to mimic attention-based token drop consistently per sample
        mask = torch.bernoulli(
            torch.full((N, 1, 1, 1, H, W), keep_prob, device=x.device, dtype=x.dtype)
        )
        return x * mask
    else:
        # Unsupported rank: return unmodified
        return x


def _cross_entropy_soft(student_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets: E[-sum p_t * log_softmax(z_s)]."""
    log_q = F.log_softmax(student_logits, dim=1)
    loss = -(target_probs * log_q).sum(dim=1).mean()
    return loss


@torch.enable_grad()
def forward_and_adapt(x: torch.Tensor, model, optimizer, args=None, actual_bz: Optional[int] = None, n_clips: Optional[int] = None):
    """
    REM forward-and-adapt on a batch.

    - Generates masked views at ratios {0, 0.05, 0.10}
    - Computes MCL and ERL losses
    - Updates only normalization layers (as configured in setup)

    Returns: (outputs_unmasked, rem_loss_scalar, mcl_scalar, erl_scalar)
    outputs_unmasked follows the same shape expectations as TENT for downstream accuracy computation.
    """
    arch = getattr(args, 'arch', None)

    # Ensure shapes for masking (permute TANet to N,C,T,H,W; leave VideoSwin/others unchanged)
    x_for_mask, inv_perm = _ensure_ncthw(x, arch)

    # Create masked variants at multiple ratios
    # Use gentler masking for TANet to avoid excessive input corruption
    ratios = [0.0, 0.02, 0.05] if arch == 'tanet' else [0.0, 0.05, 0.10]
    xs = [_apply_spatial_mask(x_for_mask, r) for r in ratios]

    # Convert back to model-required shapes per arch
    xs_model = []
    for t in xs:
        if arch == 'tanet':
            xs_model.append(_invert_to_tanet(t, inv_perm))
        else:
            xs_model.append(t)

    # Forward passes
    logits_list = []
    for xin in xs_model:
        outputs = model(xin)
        logits_list.append(outputs)

    # For TANet, average over views like TENT does inside its forward
    if arch == 'tanet' and actual_bz is not None and n_clips is not None:
        proc_logits = []
        for out in logits_list:
            out = out.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
            proc_logits.append(out)
    else:
        proc_logits = logits_list

    # Extract logits tensor if model returns tuple/dict (non-TANet path common in this repo)
    def _to_logits(o):
        if isinstance(o, (tuple, list)):
            return o[0]
        if isinstance(o, dict):
            return o.get('logits', next(iter(o.values())))
        return o

    proc_logits = [ _to_logits(o) for o in proc_logits ]

    # Unmasked prediction for accuracy/reporting
    logits0 = proc_logits[0]

    # Build MCL (masked consistency): sum CE( z_j, sg(softmax(z_i)) ) for i<j
    with torch.no_grad():
        targets_probs = [ F.softmax(l, dim=1) for l in proc_logits ]
    mcl = _cross_entropy_soft(proc_logits[1], targets_probs[0]) \
        + _cross_entropy_soft(proc_logits[2], targets_probs[0]) \
        + _cross_entropy_soft(proc_logits[2], targets_probs[1])

    # Build ERL (entropy ranking hinge): sum ReLU( S(z_i) - sg(S(z_j)) + m ) for i<j
    ent = [ softmax_entropy(l) for l in proc_logits ]  # per-sample entropy
    m = 0.0
    erl = F.relu(ent[0] - ent[1].detach() + m).mean() \
        + F.relu(ent[0] - ent[2].detach() + m).mean() \
        + F.relu(ent[1] - ent[2].detach() + m).mean()

    lam = 1.0
    loss = mcl + lam * erl

    loss.backward()
    # Gradient clipping to stabilize TANet updates
    try:
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)
    except Exception:
        pass
    optimizer.step()
    optimizer.zero_grad()

    return logits0, float(loss.item()), float(mcl.item()), float(erl.item())


def configure_model(model):
    """Configure model for REM (same as TENT): update only BN affine params."""
    return tent.configure_model(model)


def collect_params(model):
    """Collect BN affine params (same as TENT)."""
    return tent.collect_params(model)
