import torch
import numpy as np
import pywt
from functools import partial


def apply_dwt_preprocessing(video_tensor, component='LL'):
    """
    Apply Discrete Wavelet Transform to video tensor and reconstruct using selected component(s).
    
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
    
    Returns:
        video_tensor with same shape but containing DWT-based processed information
    """
    # Debug: log input tensor info before any conversion
    print(f"[DWT DEBUG] input: type={type(video_tensor)}, device={getattr(video_tensor, 'device', None)}, dtype={getattr(video_tensor, 'dtype', None)}, shape={tuple(video_tensor.shape)}")
    # Convert to numpy for DWT operations
    video_np = video_tensor.numpy()
    # Debug: log numpy view info
    print(f"[DWT DEBUG] numpy: dtype={video_np.dtype}, shape={video_np.shape}")
    
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
    # Debug: log chosen components
    print(f"[DWT DEBUG] component='{component}', parsed={components}, indices={component_indices}")
    
    # Handle different tensor shapes
    if len(shape) == 4:  # (C, T, H, W)
        C, T, H, W = shape
        print(f"[DWT DEBUG] Entering 4D branch (C,T,H,W)=({C},{T},{H},{W})")
        # Debug stop after printing
        raise RuntimeError("[DWT DEBUG] Early stop after prints (4D)")
        # Create output array
        processed_video = np.zeros_like(video_np)
        
        # Process each channel and time frame
        for c in range(video_np.shape[0]):  # Channels
            for t in range(video_np.shape[1]):  # Time frames
                # Apply 2D DWT to each frame
                coeffs2 = pywt.dwt2(video_np[c, t], 'haar', mode='smooth')
                
                # Extract components
                cA, (cH, cV, cD) = coeffs2
                
                # Zero out unselected components and keep only the selected ones
                if 0 not in component_indices:  # LL
                    cA = np.zeros_like(cA)
                if 1 not in component_indices:  # LH
                    cH = np.zeros_like(cH)
                if 2 not in component_indices:  # HL
                    cV = np.zeros_like(cV)
                if 3 not in component_indices:  # HH
                    cD = np.zeros_like(cD)
                
                # Apply inverse 2D DWT
                reconstructed_frame = pywt.idwt2((cA, (cH, cV, cD)), 'haar', mode='smooth')
                
                # Handle shape mismatch due to DWT padding
                if reconstructed_frame.shape != video_np[c, t].shape:
                    # Crop to original size
                    reconstructed_frame = reconstructed_frame[:video_np[c, t].shape[0], :video_np[c, t].shape[1]]
                
                processed_video[c, t] = reconstructed_frame
        
        # Convert back to tensor
        return torch.from_numpy(processed_video).to(video_tensor.device)
    elif len(shape) == 3:  # (C, H, W)
        C, H, W = shape
        print(f"[DWT DEBUG] Entering 3D branch (C,H,W)=({C},{H},{W})")
        # Debug stop after printing
        raise RuntimeError("[DWT DEBUG] Early stop after prints (3D)")
        # Create output array
        processed_video = np.zeros_like(video_np)
        
        # Process each channel
        for c in range(video_np.shape[0]):  # Channels
            # Apply 2D DWT to the image
            coeffs2 = pywt.dwt2(video_np[c], 'haar', mode='smooth')
            
            # Extract components
            cA, (cH, cV, cD) = coeffs2
            
            # Zero out unselected components and keep only the selected ones
            if 0 not in component_indices:  # LL
                cA = np.zeros_like(cA)
            if 1 not in component_indices:  # LH
                cH = np.zeros_like(cH)
            if 2 not in component_indices:  # HL
                cV = np.zeros_like(cV)
            if 3 not in component_indices:  # HH
                cD = np.zeros_like(cD)
            
            # Apply inverse 2D DWT
            reconstructed_frame = pywt.idwt2((cA, (cH, cV, cD)), 'haar', mode='smooth')
            
            # Handle shape mismatch due to DWT padding
            if reconstructed_frame.shape != video_np[c].shape:
                # Crop to original size
                reconstructed_frame = reconstructed_frame[:video_np[c].shape[0], :video_np[c].shape[1]]
            
            processed_video[c] = reconstructed_frame
    
        # Convert back to tensor
        return torch.from_numpy(processed_video).to(video_tensor.device)
    else:
        print(f"[DWT DEBUG] Unsupported tensor ndim={len(shape)}, shape={shape}")
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


def apply_phase_only_preprocessing_batch(video_batch):
    """
    Apply phase-only preprocessing to a batch of videos.
    
    Args:
        video_batch: Tensor of shape (B, C, T, H, W) where B is batch size
    
    Returns:
        video_batch with same shape but containing only phase information
    """
    B, C, T, H, W = video_batch.shape
    phase_only_batch = torch.zeros_like(video_batch)
    
    for b in range(B):
        phase_only_batch[b] = apply_phase_only_preprocessing(video_batch[b])
    
    return phase_only_batch
