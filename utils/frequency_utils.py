import torch
import numpy as np
import pywt
from functools import partial


def apply_dwt_preprocessing(video_tensor, component='LL', levels: int = 1):
    """
    Apply multi-level Discrete Wavelet Transform to video tensor and reconstruct using selected component(s).
    
    The DWT decomposes an image into four subbands:
    - 'LL': Low-Low - Approximation coefficients (most important for overall structure)
    - 'LH': Low-High - Horizontal detail coefficients (horizontal edges)
    - 'HL': High-Low - Vertical detail coefficients (vertical edges)
    - 'HH': High-High - Diagonal detail coefficients (diagonal edges)
    
    You can also combine components by separating them with '+':
    - 'LL+LH': Combine approximation and horizontal detail
    - 'LL+HL': Combine approximation and vertical detail
    - 'LL+HH': Combine approximation and diagonal detail
    
    Args:
        video_tensor: Tensor of shape (C, T, H, W) or (C, H, W) where C is channels, T is time, H is height, W is width
        component: Which component(s) to use for reconstruction ('LL', 'LH', 'HL', 'HH' or combinations like 'LL+LH')
        levels: Number of decomposition levels K (>=1). Components are applied at the deepest level.
    
    Returns:
        video_tensor with same shape but containing DWT-based processed information
    """
    # Convert to numpy for DWT operations
    video_np = video_tensor.detach().cpu().numpy()
    
    # Get shape
    shape = video_np.shape
    
    # Parse components (support for combinations like 'LL+LH')
    components = component.split('+')
    
    # Validate components
    valid_components = ['LL', 'LH', 'HL', 'HH']
    for comp in components:
        if comp not in valid_components:
            raise ValueError(f"Invalid component '{comp}'. Must be one of {valid_components} or combinations like 'LL+LH'")
    
    # Map component names to indices
    component_map = {
        'LL': 0,  # approx
        'LH': 1,  # horizontal
        'HL': 2,  # vertical
        'HH': 3   # diagonal
    }
    component_indices = [component_map[comp] for comp in components]
    
    # Helper to process a single 2D frame with K-level DWT
    def _process_frame_2d(frame_2d: np.ndarray) -> np.ndarray:
        # Clamp levels to the maximum allowed by the frame size and wavelet filter length
        try:
            w = pywt.Wavelet('haar')
            max_h = pywt.dwt_max_level(frame_2d.shape[0], w.dec_len)
            max_w = pywt.dwt_max_level(frame_2d.shape[1], w.dec_len)
            max_lvl = max(1, min(max_h, max_w))
        except Exception:
            max_lvl = max(1, int(levels))
        use_lvl = min(max(1, int(levels)), max_lvl)

        # Multi-level decomposition (consistent boundary mode)
        coeffs = pywt.wavedec2(frame_2d, wavelet='haar', level=use_lvl, mode='smooth')
        # coeffs structure: [cA_n, (cH_n, cV_n, cD_n), (cH_{n-1}, cV_{n-1}, cD_{n-1}), ..., (cH_1, cV_1, cD_1)]
        cA_n = coeffs[0]
        details = coeffs[1:]

        # Build modified coeffs: keep only selected components at deepest level; zero out others
        # Deepest details
        if len(details) > 0:
            cH_n, cV_n, cD_n = details[0]
        else:
            # If only approximation exists (very small images), synthesize zeros for details
            cH_n = np.zeros_like(cA_n)
            cV_n = np.zeros_like(cA_n)
            cD_n = np.zeros_like(cA_n)

        keep_cA = (0 in component_indices)
        keep_cH = (1 in component_indices)
        keep_cV = (2 in component_indices)
        keep_cD = (3 in component_indices)

        cA_n_keep = cA_n if keep_cA else np.zeros_like(cA_n)
        cH_n_keep = cH_n if keep_cH else np.zeros_like(cH_n)
        cV_n_keep = cV_n if keep_cV else np.zeros_like(cV_n)
        cD_n_keep = cD_n if keep_cD else np.zeros_like(cD_n)

        # Zero out all other levels' details
        modified_details = [(cH_n_keep, cV_n_keep, cD_n_keep)]
        for lvl in range(1, len(details)):
            dH, dV, dD = details[lvl]
            modified_details.append((np.zeros_like(dH), np.zeros_like(dV), np.zeros_like(dD)))

        coeffs_mod = [cA_n_keep] + modified_details
        recon = pywt.waverec2(coeffs_mod, wavelet='haar', mode='smooth')
        return recon

    # Handle different tensor shapes
    if len(shape) == 4:  # (C, T, H, W)
        C, T, H, W = shape
        # Create output array
        processed_video = np.zeros_like(video_np)
        
        # Process each channel and time frame
        for c in range(video_np.shape[0]):  # Channels
            for t in range(video_np.shape[1]):  # Time frames
                # Apply K-level 2D DWT to each frame and reconstruct
                reconstructed_frame = _process_frame_2d(video_np[c, t])
                
                # Handle shape mismatch due to DWT padding
                if reconstructed_frame.shape != video_np[c, t].shape:
                    # Crop to original size
                    reconstructed_frame = reconstructed_frame[:video_np[c, t].shape[0], :video_np[c, t].shape[1]]
                
                processed_video[c, t] = reconstructed_frame
        
        # Convert back to tensor
        return torch.from_numpy(processed_video).to(video_tensor.device).type_as(video_tensor)
    elif len(shape) == 3:  # (C, H, W)
        C, H, W = shape
        # Create output array
        processed_video = np.zeros_like(video_np)
        
        # Process each channel
        for c in range(video_np.shape[0]):  # Channels
            # Apply K-level 2D DWT to the image and reconstruct
            reconstructed_frame = _process_frame_2d(video_np[c])
            
            # Handle shape mismatch due to DWT padding
            if reconstructed_frame.shape != video_np[c].shape:
                # Crop to original size
                reconstructed_frame = reconstructed_frame[:video_np[c].shape[0], :video_np[c].shape[1]]
            
            processed_video[c] = reconstructed_frame
    
        # Convert back to tensor
        return torch.from_numpy(processed_video).to(video_tensor.device).type_as(video_tensor)
    else:
        raise ValueError(f"Unsupported tensor shape: {shape}")


def apply_phase_only_preprocessing(video_tensor):
    """
    Apply Fourier transform to retain only phase information.
    
    Args:
        video_tensor: Tensor of shape (C, T, H, W) or (C, H, W) where C is channels, T is time, H is height, W is width
    
    Returns:
        video_tensor with same shape but containing only phase information
    """
    # Convert to numpy for FFT operations
    video_np = video_tensor.numpy()
    
    # Get shape
    shape = video_np.shape
    
    # Handle different tensor shapes
    if len(shape) == 4:  # (C, T, H, W)
        C, T, H, W = shape
        # Create output array
        phase_only_video = np.zeros_like(video_np)
        
        # Process each channel and time frame
        for c in range(C):
            for t in range(T):
                # Apply 2D FFT
                frame_fft = np.fft.fft2(video_np[c, t, :, :])
                
                # Extract phase information (set magnitude to 1)
                phase = np.angle(frame_fft)
                
                # Reconstruct with unit magnitude
                phase_only_frame = np.real(np.fft.ifft2(np.exp(1j * phase)))
                
                # Store result
                phase_only_video[c, t, :, :] = phase_only_frame
    elif len(shape) == 3:  # (C, H, W)
        C, H, W = shape
        # Create output array
        phase_only_video = np.zeros_like(video_np)
        
        # Process each channel
        for c in range(C):
            # Apply 2D FFT
            frame_fft = np.fft.fft2(video_np[c, :, :])
            
            # Extract phase information (set magnitude to 1)
            phase = np.angle(frame_fft)
            
            # Reconstruct with unit magnitude
            phase_only_frame = np.real(np.fft.ifft2(np.exp(1j * phase)))
            
            # Store result
            phase_only_video[c, :, :] = phase_only_frame
    else:
        raise ValueError(f"Unsupported tensor shape: {shape}")
    
    # Convert back to tensor
    return torch.from_numpy(phase_only_video).float()


def apply_dwt_preprocessing_batch(video_batch, component='LL', levels: int = 1):
    """
    Apply K-level DWT preprocessing to a batch of videos.

    Supports the following input shapes:
    - (B, T, C, H, W)
    - (B, C, T, H, W)

    Returns a tensor with the same shape as input.
    """
    assert video_batch.ndim in (5,), f"Expected 5D tensor, got shape {video_batch.shape}"
    B = video_batch.shape[0]
    out = torch.zeros_like(video_batch)

    # Case 1: (B, T, C, H, W) -> per item convert to (C, T, H, W)
    if video_batch.shape[1] != 3:  # likely (B, T, C, H, W)
        for b in range(B):
            vid = video_batch[b]  # (T, C, H, W)
            # Reorder to (C, T, H, W)
            vid_cthw = vid.permute(1, 0, 2, 3).contiguous()
            processed = apply_dwt_preprocessing(vid_cthw, component=component, levels=levels)  # (C, T, H, W)
            # Back to (T, C, H, W)
            out[b] = processed.permute(1, 0, 2, 3).contiguous()
    else:
        # Case 2: (B, C, T, H, W)
        for b in range(B):
            vid = video_batch[b]  # (C, T, H, W)
            processed = apply_dwt_preprocessing(vid, component=component, levels=levels)  # (C, T, H, W)
            out[b] = processed
    return out
