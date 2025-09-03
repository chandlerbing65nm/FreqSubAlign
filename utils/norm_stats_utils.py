import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils_ import AverageMeter, AverageMeterTensor, MovingAverageTensor

# Cache for orthonormal DCT-II matrices per (N, device, dtype)
_DCT_MATS = {}

def _get_dct_matrix(N: int, device, dtype):
    """Return orthonormal DCT-II matrix of size (N,N) on device/dtype.
    C[k,n] = alpha(k) * sqrt(2/N) * cos(pi*(n+0.5)*k/N), with alpha(0)=1/sqrt(2).
    """
    key = (N, device, dtype)
    cached = _DCT_MATS.get(key)
    if cached is not None:
        return cached
    n = torch.arange(N, device=device, dtype=dtype)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    # compute cos(pi*(n+0.5)*k/N) where broadcasting gives (N, N)
    cos_arg = (torch.pi * (n + 0.5) * k) / N
    C = torch.cos(cos_arg)
    C *= (2.0 / N) ** 0.5
    C[0, :] *= 0.5 ** 0.5  # alpha(0) = 1/sqrt(2)
    _DCT_MATS[key] = C
    return C
from config import device

l1_loss = nn.L1Loss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')

def compute_kld(mean_true, mean_pred, var_true, var_pred):
    # mean1 and std1 are for true distribution
    # mean2 and std2 are for pred distribution
    # kld_mv = torch.log(std_pred / std_true) + (std_true ** 2 + (mean_true - mean_pred) ** 2) / (2 * std_pred ** 2) - 0.5

    kld_mv = 0.5 * torch.log(torch.div(var_pred, var_true)) + (var_true + (mean_true - mean_pred) ** 2) / \
             (2 * var_pred) - 0.5
    kld_mv = torch.sum(kld_mv)
    return kld_mv

class ComputeNormStatsHook():
    """
    this hook is to be placed after the normalization layer.
    """
    def __init__(self, module, clip_len = None, stat_type = None, before_norm = None, batch_size = None):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        self.stat_type = stat_type
        self.before_norm = before_norm
        self.batch_size = batch_size
    def hook_fn(self, module, input, output):

        feature = input[0] if self.before_norm else output

        if isinstance(module, nn.BatchNorm1d):
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')

            # output is in shape (N, C, T)  or   (N*C, T )
            assert self.stat_type in ['temp', 'temp_v2']
            if len(feature.size()) == 2:

                # todo should have converted  (N*C, T) to (N, C, T), this requires the ACTUAL batch size
                #    but we do not know the actual batch size
                nc, t = feature.size()
                # if self.stat_type == 'temp':
                self.batch_mean = feature.mean(0) # (N*C, T) -> (T, )
                self.batch_var = feature.permute(1, 0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                # elif self.stat_type == 'temp_v2':
                #     c = nc // self.batch_size
                #     feature = feature.view(self.batch_size, c, t)
                #     self.batch_mean = feature.mean((0, 2))  # (N, C, T) -> (C, )
                #     self.batch_var = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1,  unbiased=False)  # (N, C, T) -> (C, N, T) -> (C, )
            elif len(feature.size()) == 3:
                n, c, t = feature.size()
                self.batch_mean = feature.mean((0, 2)) # (N, C, T) -> (C, )
                self.batch_var = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )


        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):

            # todo reshape the output into (N, C, T, H, W )
            if isinstance(module, nn.BatchNorm2d):
                # output is in shape (N*T,  C,  H,  W)
                nt, c, h, w = feature.size()
                t = self.clip_len
                bz = nt // t
                feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
            elif isinstance(module, nn.BatchNorm3d):
                # output is in shape (N, C, T, H, W)
                bz, c, t, h, w = feature.size()
                feature = feature
            else:
                raise Exception(f'undefined module {module}')

            self.compute_stat_for_NCTHW(feature)

        elif isinstance(module, nn.LayerNorm):
            # todo output is in shape  B T H W C,  !!!!!!!!!!!!!!  notice that in LayerNorm, the mean and variance are computed on the C-dimension
            assert len(feature.size()) == 5
            bz, t, h, w, c = feature.size()
            feature = feature.permute(0, 4, 1, 2, 3).contiguous() #  bz, t, h, w, c ->  bz, c, t, h, w
            self.compute_stat_for_NCTHW(feature)
    def compute_stat_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()
        if self.stat_type == 'temp':
            # todo compute the statistics along N and T
            #  the statistics are in shape  (C, H, W),
            self.batch_mean = output.mean((0, 2))  # (N, C, T, H, W) ->  (C, H, W)
            # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
            self.batch_var = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False)  # (N, C, T, H, W)  ->  (C, H, W, N, T) -> (C, H, W )
        elif self.stat_type == 'temp_v2':
            output = output.mean((3,4))  # (N, C, T, H, W) -> (N, C, T)
            self.batch_mean = output.mean((0,2)) # (N, C, T) -> (C,)
            self.batch_var = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
        elif self.stat_type == 'spatiotemp':
            self.batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
            # batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1,  unbiased=False)  # compute the variance along each channel
            self.batch_var = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
        elif self.stat_type == 'spatial':
            self.batch_mean = output.mean((0, 3, 4))  # (N, C, T, H, W) ->  (C, T)
            self.batch_var = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1, unbiased=False)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T )

    def close(self):
        self.hook.remove()

class CombineNormStatsRegHook_DWT():
    """
    DWT subband statistics regularization hook.

    - Registers a forward hook on a normalization layer
    - On forward, converts features to N,C,T,H,W
    - Applies L-level 2D Haar DWT on spatial dims per frame (T) and channel (C)
    - Extracts deepest-level subbands (LL, LH, HL, HH)
    - Computes per-channel mean/var for each subband over N,T, H', W'
    - Computes weighted regularization loss vs. provided clean stats per subband
    """
    def __init__(
        self,
        module,
        clip_len: int,
        dwt_levels: int,
        clean_stats_per_band: dict,
        band_lambdas: dict,
        reg_type: str = 'mse_loss',
        moving_avg: bool = False,
        momentum: float = 0.1,
        before_norm: bool = False,
        if_sample_tta_aug_views: bool = False,
        n_augmented_views: int = 1,
        dwt_3d: bool = False,
        subband_transform: str = 'dwt',
    ):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.clip_len = clip_len
        self.dwt_levels = max(1, int(dwt_levels))
        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.before_norm = before_norm
        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.n_augmented_views = n_augmented_views
        self.dwt_3d = bool(dwt_3d)
        self.subband_transform = str(subband_transform).lower()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Expect keys: 'LL', 'LH', 'HL', 'HH' each with tuple(mean, var)
        self.bands = ['LL', 'LH', 'HL', 'HH']
        self.band_lambdas = {k: float(band_lambdas.get(k, 0.0)) for k in self.bands}

        # Initialize running totals for total and per-band weighted regularization losses
        self.r_feature = torch.tensor(0.0, device=self.device)
        self.r_feature_bands = {b: torch.tensor(0.0, device=self.device) for b in self.bands}

        self.source_mean = {}
        self.source_var = {}
        for b in self.bands:
            pair = clean_stats_per_band.get(b, (None, None)) if clean_stats_per_band is not None else (None, None)
            mu, var = pair
            if mu is not None and var is not None:
                self.source_mean[b] = torch.tensor(mu, device=self.device)
                self.source_var[b] = torch.tensor(var, device=self.device)
            else:
                self.source_mean[b] = None
                self.source_var[b] = None

        # Set up meters
        Meter = MovingAverageTensor if self.moving_avg else AverageMeterTensor
        self.mean_m = {b: Meter(momentum=self.momentum) if self.moving_avg else Meter() for b in self.bands}
        self.var_m = {b: Meter(momentum=self.momentum) if self.moving_avg else Meter() for b in self.bands}

    @staticmethod
    def _haar_kernels(device):
        # 2x2 separable Haar filters (outer products). Normalize by 2 for energy preservation.
        kLL = torch.tensor([[1., 1.], [1., 1.]], device=device) / 2.0
        kLH = torch.tensor([[1., -1.], [1., -1.]], device=device) / 2.0
        kHL = torch.tensor([[1., 1.], [-1., -1.]], device=device) / 2.0
        kHH = torch.tensor([[1., -1.], [-1., 1.]], device=device) / 2.0
        return kLL, kLH, kHL, kHH

    @staticmethod
    def _haar_kernels3d(device):
        """
        2x2x2 separable 3D Haar filters. Normalize by 2*sqrt(2) for energy preservation.
        Order: [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH] over axes (T,H,W).
        """
        import math
        L = torch.tensor([1., 1.], device=device)
        H = torch.tensor([1., -1.], device=device)
        denom = 2.0 * math.sqrt(2.0)
        def outer3(a, b, c):
            return (a[:, None, None] * b[None, :, None] * c[None, None, :]) / denom
        kLLL = outer3(L, L, L)
        kLLH = outer3(L, L, H)
        kLHL = outer3(L, H, L)
        kLHH = outer3(L, H, H)
        kHLL = outer3(H, L, L)
        kHLH = outer3(H, L, H)
        kHHL = outer3(H, H, L)
        kHHH = outer3(H, H, H)
        return kLLL, kLLH, kLHL, kLHH, kHLL, kHLH, kHHL, kHHH

    @staticmethod
    def _dwt2d_per_frame(x):
        """
        x: (N, C, H, W)
        returns tuple of subbands each (N, C, H/2, W/2)
        """
        device = x.device
        N, C, H, W = x.shape
        kLL, kLH, kHL, kHH = CombineNormStatsRegHook_DWT._haar_kernels(device)
        # Build group conv kernels of shape (4*C, 1, 2, 2) then use groups=C and repeat per channel
        base = torch.stack([kLL, kLH, kHL, kHH], dim=0)  # (4, 2, 2)
        weight = base.unsqueeze(1)  # (4,1,2,2)
        weight = weight.repeat(C, 1, 1, 1)  # (4*C,1,2,2)
        x_r = x.view(N, C, H, W)
        y = F.conv2d(x_r, weight, bias=None, stride=2, padding=0, groups=C)  # (N, 4*C, H/2, W/2)
        y = y.view(N, C, 4, y.shape[-2], y.shape[-1]).contiguous()  # (N, C, 4, H2, W2)
        LL = y[:, :, 0]
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LL, LH, HL, HH

    @staticmethod
    def _fft2d_level1(x):
        """
        2D FFT per frame on spatial dims. Partition unshifted spectrum into 4 equal quadrants.
        x: (N, C, T, H, W)
        returns LL, LH, HL, HH each (N, C, T, H//2, W//2) based on magnitude of fft2 (no shift).
        """
        N, C, T, H, W = x.shape
        # Stack frames into batch for efficiency
        xt = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
        # Complex spectrum
        # Use orthonormal FFT to keep spectrum scale stable across spatial sizes
        spec = torch.fft.fft2(xt, dim=(-2, -1), norm='ortho')
        mag = torch.abs(spec)
        h2, w2 = H // 2, W // 2
        # Quadrants (top-left is low-low due to unshifted zero-freq at [0,0])
        qLL = mag[:, :, 0:h2, 0:w2]
        qLH = mag[:, :, 0:h2, w2:]
        qHL = mag[:, :, h2:, 0:w2]
        qHH = mag[:, :, h2:, w2:]
        # Reshape back to (N, C, T, hq, wq) using dynamic quadrant sizes (supports odd H/W)
        def back(v):
            hq, wq = v.shape[-2], v.shape[-1]
            return v.view(N, T, C, hq, wq).permute(0, 2, 1, 3, 4).contiguous()
        return back(qLL), back(qLH), back(qHL), back(qHH)

    @staticmethod
    def _dct2d_level1(x):
        """
        2D DCT-II per frame (separable along H and W). Partition into equal quadrants.
        Uses torch.fft.dct if available; otherwise falls back to matmul with orthonormal DCT matrices.
        x: (N, C, T, H, W)
        returns LL, LH, HL, HH each (N, C, T, H//2, W//2)
        """
        N, C, T, H, W = x.shape
        xt = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
        if hasattr(torch.fft, 'dct'):
            # Apply DCT-II along H then W (ortho normalization for energy preservation)
            coeff = torch.fft.dct(torch.fft.dct(xt, type=2, dim=-2, norm='ortho'), type=2, dim=-1, norm='ortho')
        else:
            # Fallback using separable orthonormal DCT matrices
            Dh = _get_dct_matrix(H, xt.device, xt.dtype)  # (H,H)
            Dw = _get_dct_matrix(W, xt.device, xt.dtype)  # (W,W)
            X = xt.reshape(-1, H, W)  # (NT*C, H, W)
            Y = torch.matmul(Dh, X)   # (NT*C, H, W)
            coeff = torch.matmul(Y, Dw.t()).reshape(N * T, C, H, W)
        h2, w2 = H // 2, W // 2
        cLL = coeff[:, :, 0:h2, 0:w2]
        cLH = coeff[:, :, 0:h2, w2:]
        cHL = coeff[:, :, h2:, 0:w2]
        cHH = coeff[:, :, h2:, w2:]
        def back(v):
            hq, wq = v.shape[-2], v.shape[-1]
            return v.view(N, T, C, hq, wq).permute(0, 2, 1, 3, 4).contiguous()
        return back(cLL), back(cLH), back(cHL), back(cHH)

    @staticmethod
    def _dwt3d_multi_level(x, levels: int):
        """
        x: (N, C, T, H, W)
        returns deepest level 3D subbands each (N, C, T/2^L, H/2^L, W/2^L)
        Subband order and names: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
        """
        N, C, T, H, W = x.shape
        cur = x
        subbands = None
        for _ in range(levels):
            device = cur.device
            kLLL, kLLH, kLHL, kLHH, kHLL, kHLH, kHHL, kHHH = CombineNormStatsRegHook_DWT._haar_kernels3d(device)
            # Stack kernels -> (8, 2, 2, 2)
            base = torch.stack([kLLL, kLLH, kLHL, kLHH, kHLL, kHLH, kHHL, kHHH], dim=0)
            weight = base.unsqueeze(1)  # (8,1,2,2,2)
            weight = weight.repeat(C, 1, 1, 1, 1)  # (8*C,1,2,2,2)
            y = F.conv3d(cur, weight, bias=None, stride=2, padding=0, groups=C)  # (N, 8*C, T2, H2, W2)
            T2, H2, W2 = y.shape[-3], y.shape[-2], y.shape[-1]
            y = y.view(N, C, 8, T2, H2, W2).contiguous()
            LLL = y[:, :, 0]
            LLH = y[:, :, 1]
            LHL = y[:, :, 2]
            LHH = y[:, :, 3]
            HLL = y[:, :, 4]
            HLH = y[:, :, 5]
            HHL = y[:, :, 6]
            HHH = y[:, :, 7]
            subbands = (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
            # iterate only on low-low-low for deeper level
            cur = LLL
        return subbands

    @staticmethod
    def _dwt2d_multi_level(x, levels: int):
        """
        x: (N, C, T, H, W)
        returns deepest level subbands each (N, C, T, H/2^L, W/2^L)
        """
        N, C, T, H, W = x.shape
        cur = x
        LL = LH = HL = HH = None
        for _ in range(levels):
            # process per frame to avoid 3D conv; stack T into batch
            cur_2d = cur.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, cur.shape[-2], cur.shape[-1])
            LL, LH, HL, HH = CombineNormStatsRegHook_DWT._dwt2d_per_frame(cur_2d)
            H2, W2 = LL.shape[-2], LL.shape[-1]
            # reshape back to (N, C, T, H2, W2)
            LL = LL.view(N, T, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()
            LH = LH.view(N, T, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()
            HL = HL.view(N, T, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()
            HH = HH.view(N, T, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()
            # iterate only LL for deeper level
            cur = LL
        return LL, LH, HL, HH

    def hook_fn(self, module, input, output):
        feature = input[0] if self.before_norm else output
        # reset losses at each forward
        self.r_feature = torch.tensor(0.0, device=self.device)
        self.r_feature_bands = {b: torch.tensor(0.0, device=self.device) for b in self.bands}

        # Reformat to N,C,T,H,W similarly to other hooks
        if isinstance(module, nn.BatchNorm1d):
            return  # DWT on 1D not supported; skip
        elif isinstance(module, nn.BatchNorm2d):
            nmt, c, h, w = feature.size()
            t = self.clip_len
            if self.if_sample_tta_aug_views:
                m = self.n_augmented_views
                bz = nmt // (m * t)
                feat = feature.view(bz * m, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            else:
                bz = nmt // t
                feat = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        elif isinstance(module, nn.BatchNorm3d):
            bz, c, t, h, w = feature.size()
            feat = feature
        elif isinstance(module, nn.LayerNorm):
            # feature shape: (N, T, H, W, C)
            assert len(feature.size()) == 5
            bz, t, h, w, c = feature.size()
            feat = feature.permute(0, 4, 1, 2, 3).contiguous()
        else:
            return

        # Apply selected subband transform
        if self.subband_transform == 'dwt':
            # DWT: support 3D or 2D multi-level
            if self.dwt_3d:
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self._dwt3d_multi_level(feat, self.dwt_levels)
                grouped = {
                    'LL': [LLL, HLL],
                    'LH': [LLH, HLH],
                    'HL': [LHL, HHL],
                    'HH': [LHH, HHH],
                }
            else:
                LL, LH, HL, HH = self._dwt2d_multi_level(feat, self.dwt_levels)
                grouped = {'LL': [LL], 'LH': [LH], 'HL': [HL], 'HH': [HH]}
        elif self.subband_transform == 'fft':
            # FFT: only 2D level-1
            LL, LH, HL, HH = self._fft2d_level1(feat)
            grouped = {'LL': [LL], 'LH': [LH], 'HL': [HL], 'HH': [HH]}
        elif self.subband_transform == 'dct':
            # DCT: only 2D level-1
            LL, LH, HL, HH = self._dct2d_level1(feat)
            grouped = {'LL': [LL], 'LH': [LH], 'HL': [HL], 'HH': [HH]}
        else:
            raise ValueError(f"Unknown subband_transform: {self.subband_transform}")

        for b in self.bands:
            if self.source_mean[b] is None or self.source_var[b] is None:
                continue
            # Concatenate along batch dimension to aggregate stats across grouped subbands
            sb_list = [t for t in grouped[b] if t is not None]
            if len(sb_list) == 0:
                continue
            sb = torch.cat(sb_list, dim=0)
            c = sb.shape[1]
            mean_c = sb.mean(dim=(0, 2, 3, 4))
            var_c = sb.permute(1, 0, 2, 3, 4).contiguous().view(c, -1).var(1, unbiased=False)
            if self.moving_avg:
                self.mean_m[b].update(mean_c)
                self.var_m[b].update(var_c)
            else:
                self.mean_m[b].update(mean_c, n=sb.shape[0])
                self.var_m[b].update(var_c, n=sb.shape[0])
            lam = self.band_lambdas.get(b, 0.0)
            if lam > 0:
                reg_b = compute_regularization(
                    self.source_mean[b], self.mean_m[b].avg, self.source_var[b], self.var_m[b].avg, self.reg_type
                )
                self.r_feature = self.r_feature + lam * reg_b
                self.r_feature_bands[b] = lam * reg_b

    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def close(self):
        self.hook.remove()

class CombineNormStatsRegHook_onereg():
    """
    Combine regularization of several types of statistics
    todo if there are multiple views, compute the statistics on the volume of multiple views , and align statistics with the source statistics,  only one regularization
    """
    def __init__(self, module, clip_len = None,
                 spatiotemp_stats_clean_tuple = None,
                 reg_type='mse_loss', moving_avg = None, momentum=0.1, stat_type_list = None, reduce_dim = True,before_norm = None ,

                 if_sample_tta_aug_views = None, n_augmented_views = None, ):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.stat_type_list = stat_type_list
        self.reduce_dim = reduce_dim
        self.before_norm = before_norm
        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.n_augmented_views = n_augmented_views
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change

        assert self.stat_type_list == ['spatiotemp']


        # self.source_mean_temp, self.source_var_temp = temp_stats_clean_tuple
        # self.source_mean_spatial, self.source_var_spatial = spatial_stats_clean_tuple
        self.source_mean_spatiotemp, self.source_var_spatiotemp = spatiotemp_stats_clean_tuple

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.source_mean_spatiotemp is not None:
            self.source_mean_spatiotemp, self.source_var_spatiotemp = torch.tensor( self.source_mean_spatiotemp).to(self.device), torch.tensor(self.source_var_spatiotemp).to(self.device)

        if self.moving_avg:
            if 'spatiotemp' in self.stat_type_list:
                self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
        else:
            if 'spatiotemp' in self.stat_type_list:
                self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = AverageMeterTensor(), AverageMeterTensor()

    def hook_fn(self, module, input, output):
        feature = input[0] if self.before_norm else output
       
        self.r_feature = torch.tensor(0).float().to(self.device)

        if isinstance(module, nn.BatchNorm1d): # todo  on BatchNorm1d, only temporal statistics regularization
            # output is in shape (N, C, T)  or   (N*C, T )
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')
            # assert self.stat_type_list == 'temp'
            if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                if self.if_sample_tta_aug_views:
                    raise NotImplementedError('temporal statistics for regularization of multiple augmented views not implemented! ')
                else:
                    if len(feature.size()) == 2:
                        nc, t = feature.size()
                        batch_mean_temp = feature.mean(0) # (N*C, T) -> (T, )
                        batch_var_temp = feature.permute(1, 0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                        bz = nc
                        self.feature_shape = (nc, t)
                    elif len(feature.size()) == 3:
                        bz, c, t = feature.size()
                        batch_mean_temp = feature.mean((0, 2)) # (N, C, T) -> (C, )
                        batch_var_temp = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
                        self.feature_shape = (bz, c, t)
                    if self.moving_avg:
                        self.mean_avgmeter_temp.update(batch_mean_temp )
                        self.var_avgmeter_temp.update(batch_var_temp )
                    else:
                        self.mean_avgmeter_temp.update(batch_mean_temp, n= bz)
                        self.var_avgmeter_temp.update(batch_var_temp, n= bz)
                    self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp, self.mean_avgmeter_temp.avg, self.source_var_temp, self.var_avgmeter_temp.avg, self.reg_type)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d): #todo on BatchNorm2d and Batchnorm3d, all types of statistics
            if self.if_sample_tta_aug_views:
                # todo   (actual_bz * n_temporal_clips  *   clip_len,  C, 256, 256)
                if isinstance(module, nn.BatchNorm2d):
                    nmt, c, h, w = feature.size()
                    t = self.clip_len
                    m = self.n_augmented_views
                    bz = nmt // (m * t)
                    feature = feature.view(bz*m, t, c, h ,w ).permute(0, 2, 1, 3, 4).contiguous()  # ( N*M*T,  C,  H, W) -> (N*M, T, C, H, W) ->  (N*M,  C, T, H, W)
                    # feature = feature.view(bz, m, t, c, h, w).permute(0, 1, 3, 2, 4, 5).contiguous() # (N*M, T, C, H, W) -> (N, M, T, C, H, W) -> (N, M, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    nm, c, t, h, w = feature.size()
                    m = self.n_augmented_views
                    bz = nm // m
                    # feature = feature.view(bz, m, c, t, h, w)
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz*m,   c, t, h, w)
                # self.compute_reg_for_NMCTHW(feature)
                self.compute_reg_for_NCTHW(feature)
            else:
                if isinstance(module, nn.BatchNorm2d):
                    # output is in shape (N*T,  C,  H,  W)
                    nt, c, h, w = feature.size()
                    t = self.clip_len
                    bz = nt // t
                    feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    # output is in shape (N, C, T, H, W)
                    bz, c, t, h, w = feature.size()
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz, c, t, h, w)

                self.compute_reg_for_NCTHW(feature)


        elif isinstance(module, nn.LayerNorm):
            if self.if_sample_tta_aug_views:
                nm, t, h, w, c = feature.size()
                m = self.n_augmented_views
                bz = nm // m
                feature = feature.permute(0, 4, 1, 2, 3).contiguous() # nm, t, h, w, c -> nm, c,  t, h, w,
                # feature = feature.view(bz, m, t, h, w, c).permute(0, 1, 5, 2,3,4).contiguous()
                # self.compute_reg_for_NMCTHW(feature)
                self.compute_reg_for_NCTHW(feature)
            else:
                assert len(feature.size()) == 5
                bz, t, h, w, c = feature.size()
                feature = feature.permute(0, 4, 1, 2, 3).contiguous()  # bz, t, h, w, c ->  bz, c, t, h, w
                self.feature_shape = (bz, c, t, h, w)
                self.compute_reg_for_NCTHW(feature)

    def compute_reg_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()

        if 'spatiotemp' in self.stat_type_list:
            batch_mean_spatiotemp = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
            batch_var_spatiotemp = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
            if self.moving_avg:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp)
            else:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp, n=bz)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_spatiotemp,
                                                                     self.mean_avgmeter_spatiotemp.avg,
                                                                     self.source_var_spatiotemp,
                                                                     self.var_avgmeter_spatiotemp.avg, self.reg_type)
    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module

    def close(self):
        self.hook.remove()

class CombineNormStatsRegHook():
    """
    Combine regularization of several types of statistics
    todo if there are multiple views, compute the statistics on each view, and align statistics of each view with the source statistics, sum up / average the reguarlizations
    """
    def __init__(self, module, clip_len = None,
                 temp_stats_clean_tuple = None, spatial_stats_clean_tuple = None, spatiotemp_stats_clean_tuple = None,
                 reg_type='mse_loss', moving_avg = None, momentum=0.1, stat_type_list = None, reduce_dim = True,before_norm = None ,

                 if_sample_tta_aug_views = None, n_augmented_views = None, ):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.stat_type_list = stat_type_list
        self.reduce_dim = reduce_dim
        self.before_norm = before_norm
        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.n_augmented_views = n_augmented_views
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change

        self.source_mean_temp, self.source_var_temp = temp_stats_clean_tuple
        self.source_mean_spatial, self.source_var_spatial = spatial_stats_clean_tuple
        self.source_mean_spatiotemp, self.source_var_spatiotemp = spatiotemp_stats_clean_tuple

        if self.source_mean_temp is not None:
            self.source_mean_temp, self.source_var_temp = torch.tensor( self.source_mean_temp).to(device), torch.tensor(self.source_var_temp).to(device)
        if self.source_mean_spatial is not None:  # todo for BatchNorm1d layer,  there are no spatial or spatiotemporal statistics
            self.source_mean_spatial, self.source_var_spatial = torch.tensor( self.source_mean_spatial).to(device), torch.tensor(self.source_var_spatial).to(device)
        if self.source_mean_spatiotemp is not None:
            self.source_mean_spatiotemp, self.source_var_spatiotemp = torch.tensor( self.source_mean_spatiotemp).to(device), torch.tensor(self.source_var_spatiotemp).to(device)

        if self.reduce_dim:
            if self.source_mean_temp is not None:
                if len(self.source_mean_temp.size()) == 3:
                    self.source_mean_temp = self.source_mean_temp.mean((1,2)) # (C, H, W) -> (C, )
                    self.source_var_temp = self.source_var_temp.mean((1,2)) # (C, H, W) -> (C, )
            if self.source_mean_spatial is not None:
                self.source_mean_spatial = self.source_mean_spatial.mean(1) # (C, T) -> (C, )
                self.source_var_spatial = self.source_var_spatial.mean(1) # (C, T) -> (C, )


        if self.moving_avg:
            if self.if_sample_tta_aug_views:
                if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp_list = [ MovingAverageTensor(momentum=self.momentum) ] * self.n_augmented_views
                    self.var_avgmeter_temp_list = [MovingAverageTensor(momentum= self.momentum )] * self.n_augmented_views
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
                    self.var_avgmeter_spatial_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
                    self.var_avgmeter_spatiotemp_list = [MovingAverageTensor(momentum=self.momentum)] * self.n_augmented_views
            else:
                if 'temp' in self.stat_type_list  or  'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp, self.var_avgmeter_temp = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial, self.var_avgmeter_spatial = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = MovingAverageTensor(momentum=self.momentum), MovingAverageTensor(momentum=self.momentum)

        else:
            if self.if_sample_tta_aug_views:
                if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp_list = [ AverageMeterTensor() ] * self.n_augmented_views
                    self.var_avgmeter_temp_list = [AverageMeterTensor()] * self.n_augmented_views
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial_list = [AverageMeterTensor()] * self.n_augmented_views
                    self.var_avgmeter_spatial_list = [AverageMeterTensor()] * self.n_augmented_views
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp_list = [AverageMeterTensor()] * self.n_augmented_views
                    self.var_avgmeter_spatiotemp_list = [AverageMeterTensor()] * self.n_augmented_views
            else:
                if 'temp' in self.stat_type_list  or 'temp_v2' in self.stat_type_list:
                    self.mean_avgmeter_temp, self.var_avgmeter_temp = AverageMeterTensor(), AverageMeterTensor()
                if 'spatial' in self.stat_type_list:
                    self.mean_avgmeter_spatial, self.var_avgmeter_spatial = AverageMeterTensor(), AverageMeterTensor()
                if 'spatiotemp' in self.stat_type_list:
                    self.mean_avgmeter_spatiotemp, self.var_avgmeter_spatiotemp = AverageMeterTensor(), AverageMeterTensor()



    def hook_fn(self, module, input, output):
        feature = input[0] if self.before_norm else output
        self.r_feature = torch.tensor(0).float().to(device)

        if isinstance(module, nn.BatchNorm1d): # todo  on BatchNorm1d, only temporal statistics regularization
            # output is in shape (N, C, T)  or   (N*C, T )
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')
            # assert self.stat_type_list == 'temp'
            if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
                if self.if_sample_tta_aug_views:
                    raise NotImplementedError('temporal statistics for regularization of multiple augmented views not implemented! ')
                else:
                    if len(feature.size()) == 2:
                        nc, t = feature.size()
                        batch_mean_temp = feature.mean(0) # (N*C, T) -> (T, )
                        batch_var_temp = feature.permute(1, 0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                        bz = nc
                        self.feature_shape = (nc, t)
                    elif len(feature.size()) == 3:
                        bz, c, t = feature.size()
                        batch_mean_temp = feature.mean((0, 2)) # (N, C, T) -> (C, )
                        batch_var_temp = feature.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
                        self.feature_shape = (bz, c, t)
                    if self.moving_avg:
                        self.mean_avgmeter_temp.update(batch_mean_temp )
                        self.var_avgmeter_temp.update(batch_var_temp )
                    else:
                        self.mean_avgmeter_temp.update(batch_mean_temp, n= bz)
                        self.var_avgmeter_temp.update(batch_var_temp, n= bz)
                    self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp, self.mean_avgmeter_temp.avg, self.source_var_temp, self.var_avgmeter_temp.avg, self.reg_type)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d): #todo on BatchNorm2d and Batchnorm3d, all types of statistics
            if self.if_sample_tta_aug_views:
                # todo   (actual_bz * n_temporal_clips  *   clip_len,  C, 256, 256)
                if isinstance(module, nn.BatchNorm2d):
                    nmt, c, h, w = feature.size()
                    t = self.clip_len
                    m = self.n_augmented_views
                    bz = nmt // (m * t)
                    feature = feature.view(bz*m, t, c, h ,w )  # ( N*M*T,  C,  H, W) -> (N*M, T, C, H, W)
                    feature = feature.view(bz, m, t, c, h, w).permute(0, 1, 3, 2, 4, 5).contiguous() # (N*M, T, C, H, W) -> (N, M, T, C, H, W) -> (N, M, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    nm, c, t, h, w = feature.size()
                    m = self.n_augmented_views
                    bz = nm // m
                    feature = feature.view(bz, m, c, t, h, w)
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz, m,  c, t, h, w)
                self.compute_reg_for_NMCTHW(feature)
            else:
                if isinstance(module, nn.BatchNorm2d):
                    # output is in shape (N*T,  C,  H,  W)
                    nt, c, h, w = feature.size()
                    t = self.clip_len
                    bz = nt // t
                    feature = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
                elif isinstance(module, nn.BatchNorm3d):
                    # output is in shape (N, C, T, H, W)
                    bz, c, t, h, w = feature.size()
                else:
                    raise Exception(f'undefined module {module}')
                self.feature_shape = (bz, c, t, h, w)

                self.compute_reg_for_NCTHW(feature)


        elif isinstance(module, nn.LayerNorm):
            if self.if_sample_tta_aug_views:
                nm, t, h, w, c = feature.size()
                m = self.n_augmented_views
                bz = nm // m
                feature = feature.view(bz, m, t, h, w, c).permute(0, 1, 5, 2,3,4).contiguous()
                self.compute_reg_for_NMCTHW(feature)
            else:
                assert len(feature.size()) == 5
                bz, t, h, w, c = feature.size()
                feature = feature.permute(0, 4, 1, 2, 3).contiguous()  # bz, t, h, w, c ->  bz, c, t, h, w
                self.feature_shape = (bz, c, t, h, w)
                self.compute_reg_for_NCTHW(feature)

    def compute_reg_for_NMCTHW(self, output):
        # todo M is the number of augmented views
        bz, m, c, t, h, w = output.size()
        if 'temp' in self.stat_type_list or 'temp_v2' in self.stat_type_list:
            raise Exception('regularization of temporal statistics not implemented')
        if 'spatial' in self.stat_type_list:
            raise Exception('regularization of temporal statistics not implemented')
        if 'spatiotemp' in self.stat_type_list:
            batch_mean_spatiotemp = output.mean((0, 3, 4, 5)) # (N, M, C, T, H, W) -> (M, C)
            batch_var_spatiotemp = output.permute(1, 2, 0, 3, 4, 5).contiguous().view([m, c, -1]).var(2, unbiased = False)  #  (N, M, C, T, H, W) -> ( M, C, N, T, H, W ) -> (M, C)
            if self.moving_avg:
                for idx in range(self.n_augmented_views):
                    self.mean_avgmeter_spatiotemp_list[idx].update(batch_mean_spatiotemp[idx, :])
                    self.var_avgmeter_spatiotemp_list[idx].update(batch_var_spatiotemp[idx, :])
            else:
                for idx in range(self.n_augmented_views):
                    self.mean_avgmeter_spatiotemp_list[idx].update(batch_mean_spatiotemp[idx, :], n=bz)
                    self.var_avgmeter_spatiotemp_list[idx].update(batch_var_spatiotemp[idx, :], n= bz)
            reg_sum = torch.tensor(0).float().to(device)
            for idx in range(self.n_augmented_views):
                reg_sum = reg_sum + compute_regularization(self.source_mean_spatiotemp, self.mean_avgmeter_spatiotemp_list[idx].avg,
                                                                         self.source_var_spatiotemp, self.var_avgmeter_spatiotemp_list[idx].avg,  self.reg_type)
            reg_sum = reg_sum / self.n_augmented_views
            self.r_feature = self.r_feature + reg_sum


    def compute_reg_for_NCTHW(self, output):
        bz, c, t, h, w = output.size()

        if 'temp' in self.stat_type_list:
            if self.reduce_dim:
                batch_mean_temp = output.mean((0, 2, 3, 4))  # (N, C, T, H, W)-> (C,)
                batch_var_temp = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False).mean( (1, 2))  # (N, C, T, H, W)-> (C, H, W, N, T) -> (C, H, W )->(C,)
            else:
                batch_mean_temp = output.mean((0, 2))  # (N, C, T, H, W) ->  (C, H, W)
                batch_var_temp = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False)  # (N, C, T, H, W)  ->  (C, H, W, N, T) -> (C, H, W )
            if self.moving_avg:
                self.mean_avgmeter_temp.update(batch_mean_temp)
                self.var_avgmeter_temp.update(batch_var_temp)
            else:
                self.mean_avgmeter_temp.update(batch_mean_temp, n=bz)
                self.var_avgmeter_temp.update(batch_var_temp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp,
                                                                     self.mean_avgmeter_temp.avg,
                                                                     self.source_var_temp,
                                                                     self.var_avgmeter_temp.avg, self.reg_type)
        if 'temp_v2' in self.stat_type_list:
            output = output.mean((3, 4))  # (N, C, T, H, W) -> (N, C, T)
            batch_mean_temp = output.mean((0, 2))  # (N, C, T) -> (C,)
            batch_var_temp = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1,  unbiased=False)  # (N, C, T) -> (C, N, T) -> (C, )
            if self.moving_avg:
                self.mean_avgmeter_temp.update(batch_mean_temp)
                self.var_avgmeter_temp.update(batch_var_temp)
            else:
                self.mean_avgmeter_temp.update(batch_mean_temp, n=bz)
                self.var_avgmeter_temp.update(batch_var_temp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_temp,
                                                                     self.mean_avgmeter_temp.avg,
                                                                     self.source_var_temp,
                                                                     self.var_avgmeter_temp.avg, self.reg_type)


        if 'spatiotemp' in self.stat_type_list:
            batch_mean_spatiotemp = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
            batch_var_spatiotemp = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
            if self.moving_avg:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp)
            else:
                self.mean_avgmeter_spatiotemp.update(batch_mean_spatiotemp, n=bz)
                self.var_avgmeter_spatiotemp.update(batch_var_spatiotemp, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_spatiotemp,
                                                                     self.mean_avgmeter_spatiotemp.avg,
                                                                     self.source_var_spatiotemp,
                                                                     self.var_avgmeter_spatiotemp.avg, self.reg_type)

        if 'spatial' in self.stat_type_list:
            if self.reduce_dim:
                batch_mean_spatial = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
                batch_var_spatial = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1,
                                                                                                    unbiased=False).mean(
                    1)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T ) -> (C, )
            else:
                batch_mean_spatial = output.mean((0, 3, 4))  # (N, C, T, H, W) ->  (C, T)
                batch_var_spatial = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1,
                                                                                                    unbiased=False)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T )
            if self.moving_avg:
                self.mean_avgmeter_spatial.update(batch_mean_spatial)
                self.var_avgmeter_spatial.update(batch_var_spatial)
            else:
                self.mean_avgmeter_spatial.update(batch_mean_spatial, n=bz)
                self.var_avgmeter_spatial.update(batch_var_spatial, n=bz)
            self.r_feature = self.r_feature + compute_regularization(self.source_mean_spatial,
                                                                     self.mean_avgmeter_spatial.avg,
                                                                     self.source_var_spatial,
                                                                     self.var_avgmeter_spatial.avg, self.reg_type)

    def close(self):
        self.hook.remove()


def compute_regularization(mean_true, mean_pred, var_true, var_pred, reg_type):
    # device = torch.device("cuda:0")
    # mean_true = mean_true.to(device)
    mean_pred = mean_pred.to(mean_true.device)
    # var_true = var_true.to(device)
    var_pred = var_pred.to(var_true.device)
    if reg_type == 'mse_loss':
        return mse_loss(var_true, var_pred) + mse_loss(mean_true, mean_pred)
    elif reg_type == 'l1_loss':
        return l1_loss(var_true, var_pred) + l1_loss(mean_true, mean_pred)
    elif reg_type == 'kld':
        return compute_kld(mean_true, mean_pred, var_true, var_pred)


class NormStatsRegHook():
    """
    Regularization of one type of statistics
    todo to be deprecated
    """
    def __init__(self, module, clip_len = None, stats_clean_tuple = None, reg_type='mse_loss', moving_avg = None, momentum=0.1, stat_type = None, reduce_dim = True):
        raise NotImplementedError('args.stat_type of str  is deprecated, use list instead. To add the implementation for case of Video swin transformer. ')

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        self.moving_avg = moving_avg
        self.momentum = momentum
        self.stat_type = stat_type
        self.reduce_dim = reduce_dim
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change

        self.source_mean, self.source_var = stats_clean_tuple

        self.source_mean = torch.tensor(self.source_mean).to(device)
        self.source_var = torch.tensor(self.source_var).to(device)
        if self.stat_type == 'temp':
            if self.reduce_dim and len(self.source_mean.size())==3 :
                self.source_mean = self.source_mean.mean((1,2)) # (C, H, W) -> (C, )
                self.source_var = self.source_var.mean((1,2 )) # (C, H, W) -> (C, )
        elif self.stat_type == 'spatial':
            if self.reduce_dim:
                self.source_mean = self.source_mean.mean(1) # (C, T) -> (C, )
                self.source_var = self.source_var.mean(1) # (C, T) -> (C, )

        if self.moving_avg:
            self.mean_avgmeter = MovingAverageTensor(momentum=self.momentum)
            self.var_avgmeter = MovingAverageTensor(momentum=self.momentum)
        else:
            self.mean_avgmeter = AverageMeterTensor()
            self.var_avgmeter = AverageMeterTensor()

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.BatchNorm1d):
            # output is in shape (N, C, T)  or   (N*C, T )
            # raise NotImplementedError('Statistics computation for nn.BatchNorm1d not implemented! ')
            assert self.stat_type == 'temp'
            if len(output.size()) == 2:
                nc, t = output.size()
                batch_mean = output.mean(0) # (N*C, T) -> (T, )
                batch_var = output.permute(1,0).contiguous().var(1, unbiased = False) # (N*C, T) -> (T, NC ) -> (T, )
                bz = nc
            elif len(output.size()) == 3:
                bz, c, t = output.size()
                batch_mean = output.mean( (0, 2)) # (N, C, T) -> (C, )
                batch_var = output.permute(1, 0, 2).contiguous().view([c, -1]).var(1, unbiased = False) # (N, C, T) -> (C, N, T) -> (C, )
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            if isinstance(module, nn.BatchNorm2d):
                # output is in shape (N*T,  C,  H,  W)
                nt, c, h, w = output.size()
                t = self.clip_len
                bz = nt // t
                output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
            elif isinstance(module, nn.BatchNorm3d):
                # output is in shape (N, C, T, H, W)
                bz, c, t, h, w = output.size()
                output = output
            else:
                raise Exception(f'undefined module {module}')

            # todo compute the batch statistics
            if self.stat_type == 'temp':
                if self.reduce_dim:
                    batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W)-> (C,)
                    batch_var = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1, unbiased=False).mean((1,2))  # (N, C, T, H, W)-> (C, H, W, N, T) -> (C, H, W )->(C,)
                else:
                    batch_mean = output.mean((0, 2))  # (N, C, T, H, W) ->  (C, H, W)
                    batch_var  = output.permute(1, 3, 4, 0, 2).contiguous().view([c, h, w, -1]).var(-1,unbiased=False)  # (N, C, T, H, W)  ->  (C, H, W, N, T) -> (C, H, W )
            elif self.stat_type == 'spatiotemp':
                batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
                batch_var = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False)  # (N, C, T, H, W)  ->  (C, N, T, H, W) -> (C, )
            elif self.stat_type == 'spatial':
                if self.reduce_dim:
                    batch_mean = output.mean((0, 2, 3, 4))  # (N, C, T, H, W) ->  (C, )
                    batch_var = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t, -1]).var(-1, unbiased=False).mean(1)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T ) -> (C, )
                else:
                    batch_mean = output.mean((0, 3, 4))  # (N, C, T, H, W) ->  (C, T)
                    batch_var = output.permute(1, 2, 0, 3, 4).contiguous().view([c, t,  -1]).var(-1,  unbiased=False)  # (N, C, T, H, W)  ->  (C, T, N, H, W ) -> (C, T )

        if self.moving_avg:
            self.mean_avgmeter.update(batch_mean)
            self.var_avgmeter.update(batch_var)
        else:
            self.mean_avgmeter.update(batch_mean, n= bz)
            self.var_avgmeter.update(batch_var, n= bz)

        if self.reg_type == 'mse_loss':
            # # todo sum of squared difference,  averaged over  h * w
            # self.r_feature = torch.sum(( self.source_var - self.var_avgmeter.avg )**2 ) / spatial_dim + torch.sum(( self.source_mean - self.mean_avgmeter.avg )**2 ) / spatial_dim
            # self.r_feature = torch.norm(self.source_var - self.var_avgmeter.avg, 2) + torch.norm(self.source_mean - self.mean_avgmeter.avg, 2)
            self.r_feature = mse_loss(self.source_var, self.var_avgmeter.avg) + mse_loss(self.source_mean, self.mean_avgmeter.avg)
        elif self.reg_type == 'l1_loss':
            self.r_feature = l1_loss(self.source_var, self.var_avgmeter.avg) + l1_loss(self.source_mean, self.mean_avgmeter.avg)
        else:
            raise NotImplementedError

    def close(self):
        self.hook.remove()