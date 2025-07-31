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

