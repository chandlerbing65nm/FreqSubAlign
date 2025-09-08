import torch
import torch.nn as nn
from utils.utils_ import AverageMeter, AverageMeterTensor, MovingAverageTensor
from utils.norm_stats_utils import compute_regularization, CombineNormStatsRegHook_DWT

l1_loss = nn.L1Loss(reduction='mean')

def compute_kld(mean_true, mean_pred, var_true, var_pred):
    # mean1 and std1 are for true distribution
    # mean2 and std2 are for pred distribution
    # kld_mv = torch.log(std_pred / std_true) + (std_true ** 2 + (mean_true - mean_pred) ** 2) / (2 * std_pred ** 2) - 0.5

    kld_mv = 0.5 * torch.log(torch.div(var_pred, var_true)) + (var_true + (mean_true - mean_pred) ** 2) / \
             (2 * var_pred) - 0.5
    kld_mv = torch.sum(kld_mv)
    return kld_mv


class BNFeatureHook():
    def __init__(self, module, reg_type='l2norm', running_manner = False, use_src_stat_in_reg = True, momentum = 0.1):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.reg_type = reg_type
        self.running_manner = running_manner
        self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change
        if self.use_src_stat_in_reg:
            self.source_mean = module.running_mean.data
            self.source_var = module.running_var.data
        if self.running_manner:
            # initialize the statistics of computation in running manner
            self.mean = torch.zeros_like( module.running_mean)
            self.var = torch.zeros_like(module.running_var)
        self.momentum = momentum

    def hook_fn(self, module, input, output):  # input in shape (B, C, T, H, W)

        nch = input[0].shape[1]
        if isinstance(module, nn.BatchNorm1d):
            # input in shape (B, C) or (B, C, T)
            if len(input[0].shape) == 2: #  todo  BatchNorm1d in TAM G branch  input is (N*C,  T )
                batch_mean = input[0].mean([0,])
                batch_var = input[0].permute(1, 0,).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
            elif len(input[0].shape) == 3:  # todo BatchNorm1d in TAM L branch  input is (N, C, T)
                batch_mean = input[0].mean([0,2])
                batch_var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
        elif isinstance(module, nn.BatchNorm2d):
            # input in shape (B, C, H, W)
            batch_mean = input[0].mean([0, 2, 3])
            batch_var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel
        elif isinstance(module, nn.BatchNorm3d):
            # input in shape (B, C, T, H, W)
            batch_mean = input[0].mean([0, 2, 3, 4])
            batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1,  unbiased=False)  # compute the variance along each channel

        self.mean =  self.momentum * batch_mean + (1.0 - self.momentum) * self.mean.detach() if self.running_manner else batch_mean
        self.var = self.momentum * batch_var + (1.0 - self.momentum) * self.var.detach() if self.running_manner else batch_var
        # todo if BN layer is set to eval, these two are the same;  otherwise, module.running_xx.data keeps changing
        self.mean_true = self.source_mean if self.use_src_stat_in_reg else module.running_mean.data
        self.var_true = self.source_var if self.use_src_stat_in_reg else module.running_var.data
        self.r_feature = compute_regularization(mean_true = self.mean_true, mean_pred = self.mean, var_true=self.var_true, var_pred = self.var, reg_type = self.reg_type)


        # if self.reg_type == 'l2norm':
        #     self.r_feature = torch.norm(self.var_true - self.var, 2) + torch.norm(self.mean_true - self.mean,2)
        # if self.reg_type == 'l1_loss':
        #     self.r_feature = torch.norm(self.var_true  - self.var, 1) + torch.norm(self.mean_true - self.mean, 1)
        # elif self.reg_type == 'kld':
        #     self.r_feature = compute_kld(mean_true=self.mean_true, mean_pred= self.mean,
        #                                             var_true= self.var_true, var_pred= self.var)

    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module

    def close(self):
        self.hook.remove()







class BNSubbandDWTRegHook:
    """
    DWT subband regularization using BN stored stats as targets.

    - Works on BatchNorm2d/BatchNorm3d (skips BatchNorm1d)
    - Computes per-channel mean/var over DWT subbands (LL/LH/HL/HH)
    - Targets are BN running stats (frozen at init if use_src_stat_in_reg=True,
      otherwise read from module.running_* at each forward)
    - Supports 2D multi-level and 3D level-wise DWT with selectable wavelet
    - Exposes r_feature and r_feature_bands for integration with existing loss code
    - Supports "running_manner" so statistics can be accumulated with EMA like BNFeatureHook
    """
    def __init__(
        self,
        module: nn.Module,
        clip_len: int,
        dwt_levels: int = 1,
        band_lambdas: dict = None,
        reg_type: str = 'mse_loss',
        moving_avg: bool = False,
        momentum: float = 0.1,
        before_norm: bool = False,
        if_sample_tta_aug_views: bool = False,
        n_augmented_views: int = 1,
        dwt_3d: bool = False,
        wavelet: str = 'haar',
        use_src_stat_in_reg: bool = True,
        running_manner: bool = False,
    ):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.clip_len = int(clip_len)
        self.dwt_levels = max(1, int(dwt_levels))
        self.reg_type = reg_type
        # running_manner behaves like an alias for moving average accumulation
        self.running_manner = bool(running_manner)
        self.moving_avg = bool(moving_avg) or self.running_manner
        self.momentum = float(momentum)
        self.before_norm = before_norm
        self.if_sample_tta_aug_views = if_sample_tta_aug_views
        self.n_augmented_views = int(n_augmented_views)
        self.dwt_3d = bool(dwt_3d)
        self.wavelet = str(wavelet).lower()
        self.use_src_stat_in_reg = bool(use_src_stat_in_reg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # BN targets: capture source stats at construction if requested
        if self.use_src_stat_in_reg and hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
            self.source_mean = module.running_mean.data.clone().to(self.device)
            self.source_var = module.running_var.data.clone().to(self.device)
        else:
            self.source_mean = None
            self.source_var = None

        # Band config
        self.bands = ['LL', 'LH', 'HL', 'HH']
        band_lambdas = band_lambdas or {}
        self.band_lambdas = {k: float(band_lambdas.get(k, 0.0)) for k in self.bands}

        # Running-EMA buffers for predicted stats per band (C-dim)
        self.mean_pred = {b: None for b in self.bands}
        self.var_pred = {b: None for b in self.bands}
        if self.running_manner:
            # Initialize zeros like BN stats (per-channel)
            template = self.source_mean if self.source_mean is not None else torch.tensor(0.0, device=self.device)
            for b in self.bands:
                if isinstance(template, torch.Tensor) and template.ndim >= 1:
                    self.mean_pred[b] = torch.zeros_like(template)
                    self.var_pred[b] = torch.zeros_like(template)
                else:
                    self.mean_pred[b] = None
                    self.var_pred[b] = None

        # Exposed losses
        self.r_feature = torch.tensor(0.0, device=self.device)
        self.r_feature_bands = {b: torch.tensor(0.0, device=self.device) for b in self.bands}

    def _to_ncthw(self, module: nn.Module, feature: torch.Tensor) -> torch.Tensor:
        # Convert normalized feature to (N,C,T,H,W) according to module type
        if isinstance(module, nn.BatchNorm1d):
            return None  # not supported
        elif isinstance(module, nn.BatchNorm2d):
            nt, c, h, w = feature.size()
            t = self.clip_len
            if self.if_sample_tta_aug_views:
                m = self.n_augmented_views
                bz = nt // (m * t)
                feat = feature.view(bz * m, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            else:
                bz = nt // t
                feat = feature.view(bz, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            return feat
        elif isinstance(module, nn.BatchNorm3d):
            # Already N,C,T,H,W
            return feature
        else:
            return None

    def _apply_dwt(self, x: torch.Tensor):
        # Return dict band -> list of subband tensors to aggregate
        if self.dwt_3d:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = CombineNormStatsRegHook_DWT._dwt3d_multi_level(
                x, self.dwt_levels, wavelet=self.wavelet
            )
            grouped = {
                'LL': [LLL, HLL],
                'LH': [LLH, HLH],
                'HL': [LHL, HHL],
                'HH': [LHH, HHH],
            }
        else:
            LL, LH, HL, HH = CombineNormStatsRegHook_DWT._dwt2d_multi_level(x, self.dwt_levels, wavelet=self.wavelet)
            grouped = {'LL': [LL], 'LH': [LH], 'HL': [HL], 'HH': [HH]}
        return grouped

    def hook_fn(self, module, input, output):
        feature = input[0] if self.before_norm else output
        # Reset losses
        self.r_feature = torch.tensor(0.0, device=self.device)
        self.r_feature_bands = {b: torch.tensor(0.0, device=self.device) for b in self.bands}

        # Prepare target BN stats
        if self.use_src_stat_in_reg and (self.source_mean is not None) and (self.source_var is not None):
            tgt_mean = self.source_mean
            tgt_var = self.source_var
        else:
            if not (hasattr(module, 'running_mean') and hasattr(module, 'running_var')):
                return
            tgt_mean = module.running_mean.data.to(self.device)
            tgt_var = module.running_var.data.to(self.device)

        # Convert to N,C,T,H,W
        feat = self._to_ncthw(module, feature)
        if feat is None:
            return

        # Apply DWT
        grouped = self._apply_dwt(feat)

        # Per-band computation similar to BNFeatureHook but on transformed subbands
        for b in self.bands:
            if self.band_lambdas.get(b, 0.0) <= 0:
                continue
            sb_list = [t for t in grouped[b] if t is not None]
            if len(sb_list) == 0:
                continue
            sb = torch.cat(sb_list, dim=0)  # (N*, C, T', H', W')
            c = sb.shape[1]
            batch_mean = sb.mean(dim=(0, 2, 3, 4))
            batch_var = sb.permute(1, 0, 2, 3, 4).contiguous().view(c, -1).var(1, unbiased=False)

            if self.running_manner:
                if self.mean_pred[b] is None or self.var_pred[b] is None:
                    # Initialize lazily if we couldn't at __init__ time
                    self.mean_pred[b] = torch.zeros_like(tgt_mean)
                    self.var_pred[b] = torch.zeros_like(tgt_var)
                mean_use = self.momentum * batch_mean + (1.0 - self.momentum) * self.mean_pred[b].detach()
                var_use = self.momentum * batch_var + (1.0 - self.momentum) * self.var_pred[b].detach()
                self.mean_pred[b] = mean_use
                self.var_pred[b] = var_use
            else:
                mean_use = batch_mean
                var_use = batch_var

            reg_b = compute_regularization(tgt_mean, mean_use, tgt_var, var_use, self.reg_type)
            lam = self.band_lambdas[b]
            self.r_feature = self.r_feature + lam * reg_b
            self.r_feature_bands[b] = lam * reg_b

    def add_hook_back(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def close(self):
        self.hook.remove()


class TempStatsRegHook():
    def __init__(self, module, clip_len = None, temp_stats_clean_tuple = None, reg_type='l2norm', ):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len
        # self.temp_mean_clean, self.temp_var_clean = temp_stats_clean_tuple

        self.reg_type = reg_type
        # self.running_manner = running_manner
        # self.use_src_stat_in_reg = use_src_stat_in_reg  # whether to use the source statistics in regularization loss
        # todo keep the initial module.running_xx.data (the statistics of source model)
        #   if BN layer is not set to eval,  these statistics will change
        # if self.use_src_stat_in_reg:
        #     self.source_mean = module.running_mean.data
        #     self.source_var = module.running_var.data
        self.source_mean, self.source_var = temp_stats_clean_tuple

        self.source_mean = torch.tensor(self.source_mean).cuda()
        self.source_var = torch.tensor(self.source_var).cuda()

        # self.source_mean = self.source_mean.mean((1,2))
        # self.source_var = self.source_var.mean((1,2 ))

        # if self.running_manner:
        #     # initialize the statistics of computation in running manner
        #     self.mean = torch.zeros_like( self.source_mean)
        #     self.var = torch.zeros_like( self.source_var)

        self.mean_avgmeter = AverageMeterTensor()
        self.var_avgmeter = AverageMeterTensor()

        # self.momentum = momentum

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t

            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')
        # spatial_dim = h * w
        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        batch_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        batch_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # batch_mean = output.mean(2).mean((0, 2,3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # batch_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2,3)) # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


        self.mean_avgmeter.update(batch_mean, n= bz)
        self.var_avgmeter.update(batch_var, n= bz)

        if self.reg_type == 'l2norm':
            # # todo sum of squared difference,  averaged over  h * w
            # self.r_feature = torch.sum(( self.source_var - self.var_avgmeter.avg )**2 ) / spatial_dim + torch.sum(( self.source_mean - self.mean_avgmeter.avg )**2 ) / spatial_dim
            self.r_feature = torch.norm(self.source_var - self.var_avgmeter.avg, 2) + torch.norm(self.source_mean - self.mean_avgmeter.avg, 2)
        else:
            raise NotImplementedError

    def close(self):
        self.hook.remove()




class ComputeSpatioTemporalStatisticsHook():
    def __init__(self, module, clip_len = None,):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t
            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')

        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        self.temp_mean = output.mean((0, 2,3,4)).mean(0) #  (N, C, T, H, W)  ->   (C, )
        self.temp_var = output.permute(1, 0, 2, 3, 4).contiguous().view([c, -1]).var(1, unbiased=False) #  (N, C, T, H, W) -> (C, N, T, H, W) -> (C, )

        # batch_mean = input[0].mean([0, 2, 3, 4])
        # batch_var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1, unbiased=False)  # compute the variance along each channel

        self.temp_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # self.temp_mean = output.mean(2).mean((0, 2, 3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2, 3) )   # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


    def close(self):
        self.hook.remove()


class ComputeTemporalStatisticsHook():
    def __init__(self, module, clip_len = None,):

        self.hook = module.register_forward_hook(self.hook_fn)  # register a hook func to a module
        self.clip_len = clip_len

    def hook_fn(self, module, input, output):

        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            # output is in shape (N, C, T)  or   (N*C, T )
            raise NotImplementedError('Temporal statistics computation for nn.Conv1d not implemented!')
        elif isinstance(module, nn.Conv2d):
            # output is in shape (N*T,  C,  H,  W)
            nt, c, h, w = output.size()
            t = self.clip_len
            bz = nt // t
            output = output.view(bz, t, c, h, w).permute(0, 2, 1, 3,  4).contiguous() # # ( N*T,  C,  H, W) -> (N, C, T, H, W)
        elif isinstance(module, nn.Conv3d):
            # output is in shape (N, C, T, H, W)
            bz, c, t, h, w = output.size()
            output = output
        else:
            raise Exception(f'undefined module {module}')

        # todo compute the statistics only along the temporal dimension T,  then take the average for all samples  N
        #  the statistics are in shape  (C, H, W),
        self.temp_mean = output.mean(2).mean(0)  #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C, H, W)
        # temp_var = new_output.permute(1, 3, 4, 0, 2).contiguous().view([c, t, -1]).var(2, unbiased = False )
        self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean(0)  # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C, H, W)

        # self.temp_mean = output.mean(2).mean((0, 2, 3)) #  (N, C, T, H, W)  ->  (N, C, H, W) ->  (C,)
        # self.temp_var = output.permute(0, 1, 3, 4, 2).contiguous().var(-1, unbiased=False).mean((0, 2, 3) )   # (N, C, T, H, W) -> #  (N, C, H, W, T) -> (N, C, H, W) ->  (C,)


    def close(self):
        self.hook.remove()


def choose_layers(model, candidate_layers):

    chosen_layers = []
    # choose all the BN layers
    # candidate_layers = [nn.BatchNorm1d,  nn.BatchNorm2d, nn.BatchNorm3d  ]
    counter = [0] * len(candidate_layers)
    # for m in model.modules():
    for nm, m in model.named_modules():
        for candidate_idx, candidate in enumerate(candidate_layers):
            if isinstance(m, candidate):
                counter[candidate_idx] += 1
                chosen_layers.append((nm, m))
    # for idx in range(len(candidate_layers)):
    #     print(f'Number of {candidate_layers[idx]}  : {counter[idx]}')
    return chosen_layers


def freeze_except_bn(model, bn_condidiate_layers, ):
    """
    freeze the model, except the BN layers
    :param model:
    :param bn_condidiate_layers:
    :return:
    """

    model.train()  #
    model.requires_grad_(False)
    for m in model.modules():
        for candidate in bn_condidiate_layers:
            if isinstance(m, candidate):
                m.requires_grad_(True)
    return model

def collect_bn_params(model, bn_candidate_layers):
    params = []
    names = []
    for nm, m in model.named_modules():
        for candidate in bn_candidate_layers:
            if isinstance(m, candidate):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']: # weight is scale gamma, bias is shift beta
                        params.append(p)
                        names.append( f"{nm}.{np}")
    return params, names