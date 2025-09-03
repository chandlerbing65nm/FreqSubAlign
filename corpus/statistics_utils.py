import time
# import os
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torchvision.datasets
import torchvision.models
import numpy as np

from config import device

from datasets_.dataset_deprecated import MyTSNDataset
from datasets_.video_dataset import MyTSNVideoDataset, MyVideoDataset


from models.r2plus1d import MyR2plus1d
from models import i3d
from models.i3d_incep import InceptionI3d
from models.tanet_models.tanet import TSN
from models.videoswintransformer_models.recognizer3d import Recognizer3D


from timm.models import create_model

from utils.transforms import *
from utils.utils_ import AverageMeter, accuracy,  get_augmentation
from utils.BNS_utils import BNFeatureHook, choose_layers
import baselines.tent as tent
import os.path as osp
from utils.pred_consistency_utils import compute_pred_consis
import copy as cp

# from corpus.training import train, validate, validate_brief
# from corpus.test_time_adaptation import tta_standard, test_time_adapt, evaluate_baselines
from corpus.dataset_utils import get_dataset, get_dataset_tanet, get_dataset_videoswin
# from corpus.model_utils import get_model

def compute_statistics(model = None, args=None, logger = None, log_time = None):
    # from utils.BNS_utils import ComputeTemporalStatisticsHook
    from utils.norm_stats_utils import ComputeNormStatsHook
    # todo candidate layers are conv layers
    # candidate_layers = [nn.Conv2d, nn.Conv3d]
    # chosen_conv_layers = choose_layers(model, candidate_layers)

    # candidate_layers = [nn.BatchNorm2d, nn.BatchNorm3d]
    compute_stat_hooks = []
    list_stat_mean = []
    list_stat_var = []
    if args.arch == 'tanet':
        if args.stat_type in ['temp', 'temp_v2']:
            # todo temporal statistics computed on all types of BN layers,  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
            candidate_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        elif args.stat_type  in ['spatial', 'spatiotemp'] :
            # todo spatial and spatiotemporal statistics computed only on   nn.BatchNorm2d, nn.BatchNorm3d
            candidate_layers = [nn.BatchNorm2d, nn.BatchNorm3d]
        chosen_layers = choose_layers(model, candidate_layers)
    elif args.arch == 'videoswintransformer':
        # todo   on Video Swin Transformer,
        #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
        candidate_layers = [nn.LayerNorm]
        chosen_layers = choose_layers(model, candidate_layers)
        chosen_layers = chosen_layers[1:]

    for layer_id, (layer_name, layer_) in enumerate(chosen_layers):
        compute_stat_hooks.append( ComputeNormStatsHook(layer_, clip_len= args.clip_length, stat_type=args.stat_type, before_norm= args.before_norm, batch_size=args.batch_size))
        list_stat_mean.append(AverageMeter())
        list_stat_var.append(AverageMeter())

    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips


    if args.arch == 'tanet':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, )
    elif args.arch == 'videoswintransformer':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    else:
        # I3D
        data_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    model.eval()  # fix the BNS during computation todo notice that this is already done in setup_model()
    with torch.no_grad():
        for batch_id, (input, target) in enumerate(data_loader):
            actual_bz = input.shape[0]
            input = input.to(device)
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                _ = model( input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)   -> (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                _ = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                # forward pass
                _ = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
            #
            if batch_id % 1000 == 0:
                print(f'{batch_id}/{len(data_loader)} batches completed ...')
            for hook_id, stat_hook in enumerate(compute_stat_hooks):
                list_stat_mean[hook_id].update(stat_hook.batch_mean, n= actual_bz)
                list_stat_var[hook_id].update(stat_hook.batch_var, n= actual_bz)

    for hook_id, stat_hook in enumerate(compute_stat_hooks):
        list_stat_mean[hook_id] = list_stat_mean[hook_id].avg.cpu().numpy()
        list_stat_var[hook_id] = list_stat_var[hook_id].avg.cpu().numpy()

    # Save as list of arrays to handle different shapes
    np.save( osp.join(args.result_dir,  f'list_{args.stat_type}_mean_{log_time}.npy'), np.array(list_stat_mean, dtype=object), allow_pickle=True)
    np.save( osp.join(args.result_dir, f'list_{args.stat_type}_var_{log_time}.npy'), np.array(list_stat_var, dtype=object), allow_pickle=True)

    # step1_img_ps = np.load(step1_img_ps_file, allow_pickle=True).item()


def compute_cos_similarity(model = None, args = None, log_time = None):
    from utils.relation_map_utils import ComputePairwiseSimilarityHook
    compute_stat_hooks = []
    list_cos_sim_mat = []
    if args.arch == 'tanet':
        candidate_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        # if args.stat_type in ['temp']:
        #
        # elif args.stat_type  in ['spatial', 'spatiotemp', 'channel'] :
        #     candidate_layers = [ nn.BatchNorm2d, nn.BatchNorm3d]
        chosen_layers = choose_layers(model, candidate_layers)
    elif args.arch == 'videoswintransformer':
        # todo   on Video Swin Transformer,
        #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
        candidate_layers = [nn.LayerNorm]
        chosen_layers = choose_layers(model, candidate_layers)
        chosen_layers = chosen_layers[1:]
    for layer_id, (layer_name, layer_) in enumerate(chosen_layers):
        # if args.stat_type in ['spatial', 'spatiotemp'] and layer_name in ['module.base_model.bn1']:
        #     compute_stat_hooks.append(None)
        #     list_cos_sim_mat.append(None)
        # else:
        if isinstance(layer_, nn.BatchNorm1d ) and args.stat_type  in ['spatial', 'spatiotemp', 'channel'] :
            compute_stat_hooks.append(None)
            list_cos_sim_mat.append(None)
        else:
            compute_stat_hooks.append( ComputePairwiseSimilarityHook(layer_, clip_len= args.clip_length, stat_type=args.stat_type, before_norm= args.before_norm, batch_size=args.batch_size))
            list_cos_sim_mat.append(AverageMeter())
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
    if args.arch == 'tanet':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, )
    elif args.arch == 'videoswintransformer':
        data_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    else:
        # I3D
        data_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    model.eval()  # fix the BNS during computation todo notice that this is already done in setup_model()
    with torch.no_grad():
        for batch_id, (input, target) in enumerate(data_loader):
            actual_bz = input.shape[0]
            input = input.to(device)
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                _ = model( input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)   -> (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                _ = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else: # todo I3D
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                # forward pass
                _ = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
            #
            if batch_id % 1000 == 0:
                print(f'{batch_id}/{len(data_loader)} batches completed ...')
            for hook_id, stat_hook in enumerate(compute_stat_hooks):
                get_hook_result = True
                if stat_hook is None:
                    get_hook_result = False
                else:
                    if stat_hook.sim_vec is None:
                        get_hook_result = False

                if not get_hook_result:
                    list_cos_sim_mat[hook_id] = None
                else:
                    list_cos_sim_mat[hook_id].update(stat_hook.sim_vec, n=actual_bz)

    for hook_id, stat_hook in enumerate(compute_stat_hooks):
        if list_cos_sim_mat[hook_id] is not None:
            list_cos_sim_mat[hook_id] = list_cos_sim_mat[hook_id].avg.cpu().numpy()

    np.save( osp.join(args.result_dir,  f'list_{args.stat_type}_relationmap_{log_time}.npy'), list_cos_sim_mat, allow_pickle=True)



def load_precomputed_statistics(args, n_layers):

    list_temp_mean_clean = list(np.load(args.temp_mean_clean_file, allow_pickle=True)) if 'temp' in args.stat_type or 'temp_v2' in args.stat_type else [None]* n_layers
    list_temp_var_clean = list(np.load(args.temp_var_clean_file, allow_pickle=True)) if 'temp' in args.stat_type or 'temp_v2' in args.stat_type else [None]* n_layers
    list_spatiotemp_mean_clean = list(np.load(args.spatiotemp_mean_clean_file, allow_pickle=True)) if 'spatiotemp' in args.stat_type else [None]* n_layers
    list_spatiotemp_var_clean = list(np.load(args.spatiotemp_var_clean_file, allow_pickle=True)) if 'spatiotemp' in args.stat_type else [None]* n_layers
    list_spatial_mean_clean = list(np.load(args.spatial_mean_clean_file, allow_pickle=True)) if 'spatial' in args.stat_type else [None]* n_layers
    list_spatial_var_clean = list(np.load(args.spatial_var_clean_file, allow_pickle=True)) if 'spatial' in args.stat_type else [None]* n_layers
    return list_temp_mean_clean, list_temp_var_clean, list_spatiotemp_mean_clean, list_spatiotemp_var_clean, list_spatial_mean_clean, list_spatial_var_clean


def compute_dwt_subband_statistics(model=None, args=None, logger=None, log_time=None):
    """
    Compute per-layer DWT subband mean/var (LL, LH, HL, HH) over the chosen normalization layers and save as NPZ.

    Output NPZ keys (each is a list aligned with chosen layers, dtype=object):
      - 'LL_mean', 'LL_var', 'LH_mean', 'LH_var', 'HL_mean', 'HL_var', 'HH_mean', 'HH_var'

    Layers used:
      - TANet: nn.BatchNorm2d, nn.BatchNorm3d
      - Video Swin: nn.LayerNorm (skip the first LN as elsewhere)
    """
    from utils.norm_stats_utils import CombineNormStatsRegHook_DWT

    bands = ['LL', 'LH', 'HL', 'HH']

    # Choose layers
    if args.arch == 'tanet':
        candidate_layers = [nn.BatchNorm2d, nn.BatchNorm3d]
        chosen_layers = choose_layers(model, candidate_layers)
    elif args.arch == 'videoswintransformer':
        candidate_layers = [nn.LayerNorm]
        chosen_layers = choose_layers(model, candidate_layers)
        chosen_layers = chosen_layers[1:]
    else:
        candidate_layers = [nn.BatchNorm2d, nn.BatchNorm3d]
        chosen_layers = choose_layers(model, candidate_layers)

    # Hook to compute batch subband stats (supports DWT/FFT/DCT)
    class ComputeDWTSubbandStatsHook:
        def __init__(self, module, clip_len, dwt_levels, before_norm=False,
                     if_sample_tta_aug_views=False, n_augmented_views=1,
                     subband_transform: str = 'dwt'):
            self.clip_len = clip_len
            self.dwt_levels = max(1, int(dwt_levels))
            self.before_norm = before_norm
            self.if_sample_tta_aug_views = if_sample_tta_aug_views
            self.n_augmented_views = n_augmented_views
            self.subband_transform = str(subband_transform).lower()
            self.hook = module.register_forward_hook(self.hook_fn)
            self.results = {b: {'mean': None, 'var': None} for b in bands}

        def hook_fn(self, module, inp, out):
            try:
                feature = inp[0] if self.before_norm else out
                # Reformat to N,C,T,H,W
                if isinstance(module, nn.BatchNorm1d):
                    return
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
                    bz, t, h, w, c = feature.size()
                    feat = feature.permute(0, 4, 1, 2, 3).contiguous()
                else:
                    return

                # Apply selected transform
                if self.subband_transform == 'dwt':
                    LL, LH, HL, HH = CombineNormStatsRegHook_DWT._dwt2d_multi_level(feat, self.dwt_levels)
                elif self.subband_transform == 'fft':
                    LL, LH, HL, HH = CombineNormStatsRegHook_DWT._fft2d_level1(feat)
                elif self.subband_transform == 'dct':
                    LL, LH, HL, HH = CombineNormStatsRegHook_DWT._dct2d_level1(feat)
                else:
                    raise ValueError(f"Unknown subband_transform: {self.subband_transform}")
                subband_tensors = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
                for b in bands:
                    sb = subband_tensors[b]
                    c = sb.shape[1]
                    mean_c = sb.mean(dim=(0, 2, 3, 4))  # (C,)
                    var_c = sb.permute(1, 0, 2, 3, 4).contiguous().view(c, -1).var(1, unbiased=False)
                    self.results[b]['mean'] = mean_c
                    self.results[b]['var'] = var_c
            except Exception:
                # If shape too small for the requested levels, leave results as None
                for b in bands:
                    self.results[b]['mean'] = None
                    self.results[b]['var'] = None

        def close(self):
            self.hook.remove()

    # Build hooks and meters
    compute_stat_hooks = []
    list_mean_meters = {b: [] for b in bands}
    list_var_meters = {b: [] for b in bands}
    seen = {b: [] for b in bands}

    for _, layer_ in chosen_layers:
        hook = ComputeDWTSubbandStatsHook(
            layer_,
            clip_len=args.clip_length,
            dwt_levels=getattr(args, 'dwt_align_levels', 1),
            before_norm=getattr(args, 'before_norm', False),
            if_sample_tta_aug_views=getattr(args, 'if_sample_tta_aug_views', False),
            n_augmented_views=getattr(args, 'n_augmented_views', 1),
            subband_transform=getattr(args, 'subband_transform', 'dwt'),
        )
        compute_stat_hooks.append(hook)
        for b in bands:
            list_mean_meters[b].append(AverageMeter())
            list_var_meters[b].append(AverageMeter())
            seen[b].append(False)

    # Data loader
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split('-')[-1])
        data_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
        data_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    else:
        n_clips = 1
        data_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    model.eval()
    with torch.no_grad():
        for batch_id, (inp, target) in enumerate(data_loader):
            actual_bz = inp.shape[0]
            inp = inp.to(device)
            if args.arch == 'tanet':
                inp = inp.view(-1, 3, inp.size(2), inp.size(3))
                inp = inp.view(actual_bz * args.test_crops * n_clips,
                               args.clip_length, 3, inp.size(2), inp.size(3))
                _ = model(inp)
            elif args.arch == 'videoswintransformer':
                _ = model(inp)
            else:
                inp = inp.reshape((-1,) + inp.shape[2:])
                _ = model(inp)

            if batch_id % 1000 == 0:
                print(f'{batch_id}/{len(data_loader)} batches completed ...')

            # Aggregate results from hooks
            for hook_id, stat_hook in enumerate(compute_stat_hooks):
                for b in bands:
                    mean_b = stat_hook.results[b]['mean']
                    var_b = stat_hook.results[b]['var']
                    if mean_b is not None and var_b is not None:
                        list_mean_meters[b][hook_id].update(mean_b, n=actual_bz)
                        list_var_meters[b][hook_id].update(var_b, n=actual_bz)
                        seen[b][hook_id] = True

    # Prepare lists aligned with chosen layers
    out = {f'{b}_mean': [] for b in bands}
    out.update({f'{b}_var': [] for b in bands})
    for b in bands:
        for hook_id in range(len(chosen_layers)):
            if seen[b][hook_id]:
                out[f'{b}_mean'].append(list_mean_meters[b][hook_id].avg.cpu().numpy())
                out[f'{b}_var'].append(list_var_meters[b][hook_id].avg.cpu().numpy())
            else:
                out[f'{b}_mean'].append(None)
                out[f'{b}_var'].append(None)

    # Enforce constraints for FFT/DCT: only 2D, level-1
    transform = getattr(args, 'subband_transform', 'dwt')
    if transform in ['fft', 'dct']:
        if getattr(args, 'dwt_align_levels', 1) != 1 and logger is not None:
            logger.warning(f"{transform.upper()} subband stats: forcing level-1 for saving; stats computed at requested level but saved as L1")
    # Save NPZ matching TTA loader expectations
    levels = getattr(args, 'dwt_align_levels', 1)
    prefix = f"{transform}_subband_stats_L{1 if transform in ['fft','dct'] else levels}"
    save_path = osp.join(args.result_dir, f'{prefix}_{log_time}.npz')
    np.savez(
        save_path,
        LL_mean=np.array(out['LL_mean'], dtype=object), LL_var=np.array(out['LL_var'], dtype=object),
        LH_mean=np.array(out['LH_mean'], dtype=object), LH_var=np.array(out['LH_var'], dtype=object),
        HL_mean=np.array(out['HL_mean'], dtype=object), HL_var=np.array(out['HL_var'], dtype=object),
        HH_mean=np.array(out['HH_mean'], dtype=object), HH_var=np.array(out['HH_var'], dtype=object),
    )
    if logger is not None:
        logger.debug(f'Saved {transform.upper()} subband stats to {save_path}')
    else:
        print(f'Saved {transform.upper()} subband stats to {save_path}')

