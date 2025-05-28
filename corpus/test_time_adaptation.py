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
from corpus.dataset_utils import get_dataset, get_dataset_tanet, get_dataset_videoswin
# from corpus.model_utils import get_model
# from corpus.statistics_utils import compute_statistics, compute_cos_similarity, load_precomputed_statistics

def tta_standard(model_origin, criterion, args=None, logger = None, writer =None):
    """
    todo  tta_standard: during adaptation, overfit to one sample, and evaluate on this sample right after adaptation. re-initilaize the model when the next sample comes
        tta_online: during adaptation, one gradient step per sample, and evaluate on this sample right after adaptation. do not re-initiaize the model when the next sample comes
    :param model:
    :param criterion:
    :param args:
    :param logger:
    :param writer:
    :return:
    """
    if args.if_tta_standard == 'tta_standard':
        # todo  overfit to one sample,  do not accumulate the target statistics in each forward step
        #   do not accumulate the target statistics for one sample in multiple gradient steps (multiple forward pass)
        #   do not accumulate the target statistics between different samples
        assert args.momentum_mvg == 1.0
        assert args.n_epoch_adapat == 1
    elif args.if_tta_standard == 'tta_online':
        assert args.momentum_mvg != 1.0  # todo accumulate the target statistics for different samples
        assert args.n_gradient_steps == 1 # todo one gradient step per sample (on forward pass per sample )
        assert args.n_epoch_adapat == 1
    from utils.norm_stats_utils import CombineNormStatsRegHook_onereg
    # from utils.relation_map_utils import CombineCossimRegHook

    candidate_bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    # candidate_bn_layers = [nn.BatchNorm2d, nn.BatchNorm3d]


    if args.arch == 'tanet':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        eval_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

    elif args.arch == 'videoswintransformer':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args,  split='val', dataset_type= 'tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        eval_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    # global_iter = 0

    epoch_result_list = []
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
    if args.if_sample_tta_aug_views:
        assert n_clips == 1
        n_augmented_views = args.n_augmented_views
    if_pred_consistency = args.if_pred_consistency if args.if_sample_tta_aug_views else False

    batch_time = AverageMeter()
    losses_ce = AverageMeter()
    losses_reg = AverageMeter()
    losses_consis = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    pred_concat = []
    gt_concat = []
    end = time.time()
    eval_loader_iterator = iter(eval_loader)

    # todo ############################################################
    # todo ##################################### choose layers
    # todo ############################################################
    if args.stat_reg == 'mean_var':
        assert args.stat_type == ['spatiotemp']
        list_spatiotemp_mean_clean = list(np.load(args.spatiotemp_mean_clean_file, allow_pickle=True))
        list_spatiotemp_var_clean = list(np.load(args.spatiotemp_var_clean_file, allow_pickle=True))
        if args.arch == 'tanet':
            bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            chosen_layers = choose_layers(model_origin, bn_layers)

            list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = [], []
            counter = 0
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                if isinstance(chosen_layer, nn.BatchNorm1d):
                    # at the position of Batchnorm1d, add None as placeholder in the list for spatial and spatiotemporal statistics
                    list_spatiotemp_mean_clean_new.append(None)
                    list_spatiotemp_var_clean_new.append(None)
                elif isinstance(chosen_layer, nn.BatchNorm2d) or isinstance(chosen_layer, nn.BatchNorm3d):
                    list_spatiotemp_mean_clean_new.append(list_spatiotemp_mean_clean[counter])
                    list_spatiotemp_var_clean_new.append(list_spatiotemp_var_clean[counter])
                    counter += 1

        elif args.arch == 'videoswintransformer':
            # todo   on Video Swin Transformer,
            #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
            candidate_layers = [nn.LayerNorm]
            chosen_layers = choose_layers(model_origin, candidate_layers)
            chosen_layers = chosen_layers[1:]

            list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = list_spatiotemp_mean_clean, list_spatiotemp_var_clean

        assert len(list_spatiotemp_mean_clean_new) == len(chosen_layers)

    if not hasattr(args, 'moving_avg'):
        args.moving_avg = False
    if not hasattr(args, 'momentum_mvg'):
        args.momentum_mvg = 0.1

    for batch_id, (input, target) in enumerate(tta_loader):  #

        setup_model_optimizer = False
        if args.if_tta_standard == 'tta_standard':
            setup_model_optimizer = True  #  setup model and optimizer before every sample comes
        elif args.if_tta_standard == 'tta_online':
            if batch_id == 0:
                setup_model_optimizer = True #  setup model and optimizer only before the first sample comes

        if setup_model_optimizer:
            print(f'Batch {batch_id}, initialize the model, update chosen layers, initialize hooks, intialize average meter')
            # todo ############################################################
            # todo #####################################  re-intialize the model, update chosen_layers from this new model
            # todo ############################################################
            model = cp.deepcopy(model_origin)
            # when we initialize the model, we have to re-choose the layers from it.
            if args.arch == 'tanet':
                # todo  temporal statistics are computed on  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                #      spatial statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,   not on Batchnorm1d
                #      spatiotemporal statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,  not on Batchnorm1d
                bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
                chosen_layers = choose_layers(model, bn_layers)
            elif args.arch == 'videoswintransformer':
                # todo   on Video Swin Transformer,
                #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
                candidate_layers = [nn.LayerNorm]
                chosen_layers = choose_layers(model, candidate_layers)
                chosen_layers = chosen_layers[1:]
            # todo ############################################################
            # todo ##################################### set up the optimizer
            # todo ############################################################
            if args.update_only_bn_affine:
                from utils.BNS_utils import freeze_except_bn, collect_bn_params
                if args.arch == 'tanet':
                    model = freeze_except_bn(model, bn_condidiate_layers=candidate_bn_layers)  # set only Batchnorm layers to trainable,   freeze all the other layers
                    params, param_names = collect_bn_params(model,  bn_candidate_layers=candidate_bn_layers)  # collecting gamma and beta in all Batchnorm layers
                elif args.arch == 'videoswintransformer':
                    model = freeze_except_bn(model,
                                             bn_condidiate_layers=[nn.LayerNorm])  # set only Batchnorm layers to trainable,   freeze all the other layers
                    params, param_names = collect_bn_params(model,
                                                            bn_candidate_layers=[nn.LayerNorm])  # collecting gamma and beta in all Batchnorm layers
                optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
            else:
                optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

            # todo ############################################################
            # todo ##################################### initialize hooks to the chosen layers for computing statistics, initialize average meter
            # todo ############################################################
            if args.stat_reg == 'mean_var':
                if isinstance(args.stat_type, str):
                    raise NotImplementedError(
                        'args.stat_type of str  is deprecated, use list instead. To add the implementation for case of Video swin transformer. ')
                elif isinstance(args.stat_type, list):
                    stat_reg_hooks = []
                    for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                        for block_name in args.chosen_blocks:
                            if block_name in chosen_layer_name:
                                stat_reg_hooks.append(
                                    CombineNormStatsRegHook_onereg(chosen_layer, clip_len=args.clip_length, # todo load statistcs, initialize the average meter
                                                                   spatiotemp_stats_clean_tuple=(
                                                                       list_spatiotemp_mean_clean_new[layer_id],
                                                                       list_spatiotemp_var_clean_new[layer_id]),
                                                                   reg_type=args.reg_type,
                                                                   moving_avg=args.moving_avg,
                                                                   momentum=args.momentum_mvg,
                                                                   stat_type_list=args.stat_type,
                                                                   reduce_dim=args.reduce_dim,
                                                                   before_norm=args.before_norm,
                                                                   if_sample_tta_aug_views=args.if_sample_tta_aug_views,
                                                                   n_augmented_views=args.n_augmented_views))
                                break
            elif args.stat_reg == 'BNS':
                # todo  regularization on BNS statistics
                # regularization on BNS statistics
                # bns_feature_hooks = []
                stat_reg_hooks = []
                chosen_layers = choose_layers(model, candidate_bn_layers)
                # for chosen_layer in chosen_layers:
                for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                    for block_name in args.chosen_blocks:
                        if block_name in chosen_layer_name:
                            # regularization between manually computed target batch statistics (whether or not in running manner) between source statistics
                            stat_reg_hooks.append(BNFeatureHook(chosen_layer, reg_type= args.reg_type, running_manner=args.running_manner,
                                              use_src_stat_in_reg=args.use_src_stat_in_reg, momentum=args.momentum_bns))
            else:
                raise Exception(f'undefined regularization type {args.stat_reg}')
        # todo ############################################################
        # todo ##################################### set the model to train mode,  freeze BN statistics
        # todo ############################################################
        model.train()  # BN layers are set to train mode
        if args.fix_BNS:  # fix the BNS during forward pass
            for m in model.modules():
                for candidate in candidate_bn_layers:
                    if isinstance(m, candidate):
                        m.eval()
        actual_bz = input.shape[0]
        input = input.to(device)
        target = target.to(device)
        # todo ############################################################
        # todo ##################################### reshape the input
        # todo ############################################################
        if args.arch == 'tanet':
            input = input.view(-1, 3, input.size(2), input.size(3))
            if args.if_sample_tta_aug_views:
                input = input.view(actual_bz * args.test_crops * n_augmented_views, args.clip_length, 3, input.size(2),  input.size(3))
            else:
                input = input.view(actual_bz * args.test_crops * n_clips, args.clip_length, 3, input.size(2), input.size(3))
        elif args.arch == 'videoswintransformer':
            pass
        else:
            raise NotImplementedError(f'Incorrect model type {args.arch}')
        # todo ############################################################
        # todo ##################################### train on one sample for multiple steps
        # todo ############################################################
        n_gradient_steps = args.n_gradient_steps
        for step_id in range(n_gradient_steps):
            if args.arch == 'tanet':
                output = model(input)
                if args.if_sample_tta_aug_views:
                    output = output.reshape(actual_bz, args.test_crops * n_augmented_views, -1)  # (N, n_views, n_class )
                    if if_pred_consistency:
                        loss_consis = compute_pred_consis(output)
                    output = output.mean(1)
                else:
                    output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)

            elif args.arch == 'videoswintransformer':
                if args.if_sample_tta_aug_views:
                    n_views = args.test_crops * n_augmented_views
                else:
                    n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                if args.if_sample_tta_aug_views:
                    if if_pred_consistency:
                        output, view_cls_score = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
                        loss_consis = compute_pred_consis(view_cls_score)
                else:
                    output, _ = model( input)
            else:
                raise NotImplementedError(f'Incorrect model type {args.arch}')
            loss_ce = criterion(output, target)
            loss_reg = torch.tensor(0).float().to(device)
            if args.stat_reg:
                for hook in stat_reg_hooks:
                    loss_reg += hook.r_feature.to(device)
            else:
                raise Exception(f'undefined regularization type {args.stat_reg}')
            if if_pred_consistency:
                loss = args.lambda_feature_reg*loss_reg + args.lambda_pred_consis * loss_consis
            else:
                loss = loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # global_iter += 1

        losses_ce.update(loss_ce.item(), actual_bz)
        losses_reg.update(loss_reg.item(), actual_bz)
        if if_pred_consistency:
            losses_consis.update(loss_consis.item(), actual_bz)

        # todo ############################################################
        # todo ##################################### remove all the hooks, no computation of statistics during inference
        # todo ############################################################
        if args.stat_reg:
            for stat_reg_hook in stat_reg_hooks:
                stat_reg_hook.close()
        else:
            raise Exception(f'undefined regularization type {args.stat_reg}')

        # todo ##########################################################################################
        # todo ################### Inference on the same batch ##############################################
        # todo ##########################################################################################
        with torch.no_grad():
            model.eval()
            input, target = next(eval_loader_iterator)
            input, target = input.to(device), target.to(device)
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                output = model( input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W         if  collapse in datsaet is True, then shape is  (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)
                # (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                output, _ = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else:
                raise NotImplementedError(f'Incorrect model type {args.arch}')
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            batch_time.update(time.time() - end)
            end = time.time()

        # todo ##########################################################################################
        # todo ################### In the case of tta_online, after inference, add the hooks back  ##############################################
        # todo ##########################################################################################
        if args.if_tta_standard == 'tta_online':
            hook_layer_counter = 0
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                for block_name in args.chosen_blocks:
                    if block_name in chosen_layer_name:
                        stat_reg_hooks[hook_layer_counter].add_hook_back(chosen_layer)
                        hook_layer_counter += 1
            assert hook_layer_counter == len(stat_reg_hooks)

        if args.verbose:
            logger.debug(('TTA Epoch{epoch}: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t'
                          'Loss consis {loss_consis.val:.4f} ({loss_consis.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_id, len(tta_loader), epoch=1, batch_time=batch_time, loss_reg=losses_reg, loss_consis = losses_consis,
                top1=top1, top5=top5)))

    epoch_result_list.append(top1.avg)

    # model_path = osp.join(  args.result_dir, f'{args.corruptions}.model' )
    # print(f'Saving models to {model_path}')
    #
    # torch.save( model.state_dict(), model_path )

    return epoch_result_list



def test_time_adapt(model, criterion, args=None, logger=None, writer=None):
    # test time adaptation for several epochs
    # from utils.norm_stats_utils import  CombineNormStatsRegHook
    from utils.norm_stats_utils import  CombineNormStatsRegHook_onereg
    from utils.relation_map_utils import CombineCossimRegHook
    candidate_bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    # candidate_bn_layers = [nn.BatchNorm2d, nn.BatchNorm3d]

    if args.update_only_bn_affine:
        from utils.BNS_utils import freeze_except_bn, collect_bn_params
        model = freeze_except_bn(model, bn_condidiate_layers=candidate_bn_layers) # set only Batchnorm layers to trainable,   freeze all the other layers
        params, param_names = collect_bn_params(model, bn_candidate_layers=candidate_bn_layers) #  collecting gamma and beta in all Batchnorm layers
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)


    if args.arch == 'tanet':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        eval_loader = torch.utils.data.DataLoader(
            get_dataset_tanet(args,  split='val', dataset_type='eval'),
            batch_size=args.batch_size_eval, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    elif args.arch == 'videoswintransformer':
        tta_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args,  split='val', dataset_type= 'tta'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        eval_loader = torch.utils.data.DataLoader(
            get_dataset_videoswin(args, split='val', dataset_type='eval'),
            batch_size=args.batch_size_eval, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    else:
        tta_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        eval_loader = torch.utils.data.DataLoader(
            get_dataset(args, split='val'),
            batch_size=args.batch_size_eval, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    global_iter = 0

    if args.stat_reg == 'mean_var':
        # regularization with prior of temporal statistics
        stat_reg_hooks = []
        # candidate_conv_layers = [nn.Conv2d, nn.Conv3d]

        if not hasattr(args, 'moving_avg'):
            args.moving_avg = False
        if not hasattr(args, 'momentum_mvg'):
            args.momentum_mvg = 0.1

        if isinstance(args.stat_type, str):
            """
            regularization of one type of statistics
            load one type of statistics 
            """
            raise NotImplementedError('args.stat_type of str  is deprecated, use list instead. To add the implementation for case of Video swin transformer. ')
            if args.stat_type == 'temp':
                bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            elif args.stat_type in ['spatial', 'spatiotemp' ]:
                bn_layers = [nn.BatchNorm2d, nn.BatchNorm3d]
            chosen_layers = choose_layers(model, bn_layers)
            # chosen_conv_layers = choose_layers(model, candidate_conv_layers)

            list_stat_mean_clean = list(np.load(args.stat_mean_clean_file, allow_pickle=True))
            list_stat_var_clean = list(np.load(args.stat_var_clean_file, allow_pickle=True))
            # assert len(list_temp_mean_clean) == len(chosen_conv_layers)
            assert len(list_stat_mean_clean) == len(chosen_layers)
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(  chosen_layers ):
                # if isinstance(bn_layer, nn.BatchNorm2d):
                # for nm, m in model.named_modules():
                for block_name in  args.chosen_blocks:
                    if block_name in chosen_layer_name :
                        stat_reg_hooks.append(NormStatsRegHook(chosen_layer, clip_len= args.clip_length,
                                                               stats_clean_tuple=(list_stat_mean_clean[layer_id], list_stat_var_clean[layer_id]), reg_type=args.reg_type, moving_avg= args.moving_avg, momentum= args.momentum_mvg,
                                                               stat_type=args.stat_type, reduce_dim=args.reduce_dim))
                        break
        elif isinstance(args.stat_type, list):
            """
            combination of regularizations of multiple types of statistics 
            load multiple types of statistics 
            """
            if args.arch == 'tanet':
                # todo  temporal statistics are computed on  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                #      spatial statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,   not on Batchnorm1d
                #      spatiotemporal statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,  not on Batchnorm1d
                bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
                chosen_layers = choose_layers(model, bn_layers)
                list_temp_mean_clean, list_temp_var_clean, list_spatiotemp_mean_clean, list_spatiotemp_var_clean, list_spatial_mean_clean, list_spatial_var_clean = load_precomputed_statistics(args, len(chosen_layers))
                if 'spatiotemp' in args.stat_type:
                    list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = [], []
                else:
                    list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new = [None] * len(chosen_layers), [None] * len(chosen_layers)
                if 'spatial' in args.stat_type:
                    list_spatial_mean_clean_new, list_spatial_var_clean_new = [], []
                else:
                    list_spatial_mean_clean_new, list_spatial_var_clean_new =[None] * len(chosen_layers), [None] * len(chosen_layers)
                counter = 0
                for layer_id,(chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                    if isinstance(chosen_layer, nn.BatchNorm1d):
                        # at the position of Batchnorm1d, add None as placeholder in the list for spatial and spatiotemporal statistics
                        if 'spatiotemp' in args.stat_type:
                            list_spatiotemp_mean_clean_new.append(None)
                            list_spatiotemp_var_clean_new.append(None)
                        if 'spatial' in args.stat_type:
                            list_spatial_mean_clean_new.append(None)
                            list_spatial_var_clean_new.append(None)
                    elif isinstance(chosen_layer, nn.BatchNorm2d) or isinstance(chosen_layer, nn.BatchNorm3d):
                        if 'spatiotemp' in args.stat_type:
                            list_spatiotemp_mean_clean_new.append( list_spatiotemp_mean_clean[counter] )
                            list_spatiotemp_var_clean_new.append( list_spatiotemp_var_clean[counter])
                        if 'spatial' in args.stat_type:
                            list_spatial_mean_clean_new.append(list_spatial_mean_clean[counter])
                            list_spatial_var_clean_new.append(list_spatial_var_clean[counter])

                        counter +=1
            elif args.arch == 'videoswintransformer':
                # todo   on Video Swin Transformer,
                #     statistics are computed on all LayerNorm layers (feature in shape BTHWC), except for the first LayerNorm after Conv3D (feature in shape B,combined_dim,C)
                candidate_layers = [nn.LayerNorm]
                chosen_layers = choose_layers(model, candidate_layers)
                chosen_layers = chosen_layers[1:]
                list_temp_mean_clean, list_temp_var_clean, list_spatiotemp_mean_clean, list_spatiotemp_var_clean, list_spatial_mean_clean, list_spatial_var_clean = load_precomputed_statistics( args, len(chosen_layers))
                list_spatial_mean_clean_new, list_spatial_var_clean_new= list_spatial_mean_clean, list_spatial_var_clean
                list_spatiotemp_mean_clean_new, list_spatiotemp_var_clean_new  = list_spatiotemp_mean_clean, list_spatiotemp_var_clean

            assert len(list_temp_mean_clean) == len(chosen_layers)
            for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
                for block_name in  args.chosen_blocks:
                    if block_name in chosen_layer_name :
                        stat_reg_hooks.append(CombineNormStatsRegHook_onereg(chosen_layer, clip_len = args.clip_length,
                            spatiotemp_stats_clean_tuple = (list_spatiotemp_mean_clean_new[layer_id], list_spatiotemp_var_clean_new[layer_id]),
                            reg_type=args.reg_type, moving_avg= args.moving_avg,  momentum=args.momentum_mvg,  stat_type_list = args.stat_type, reduce_dim = args.reduce_dim,
                            before_norm= args.before_norm, if_sample_tta_aug_views= args.if_sample_tta_aug_views, n_augmented_views=args.n_augmented_views ))
                        break
    elif args.stat_reg == 'cossim':
        stat_reg_hooks = []
        list_temp_cossim = list(np.load( args.temp_cossim_clean_file, allow_pickle=True ))
        if args.arch == 'tanet':
            bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            chosen_layers = choose_layers(model, bn_layers)
        assert len(list_temp_cossim) == len(chosen_layers)
        for layer_id, (chosen_layer_name, chosen_layer) in enumerate(chosen_layers):
            if list_temp_cossim[layer_id] is not None:
                for block_name in args.chosen_blocks:
                    if block_name in chosen_layer_name:
                        stat_reg_hooks.append(CombineCossimRegHook(chosen_layer, clip_len=args.clip_length,
                                                                   temp_cossim= list_temp_cossim[layer_id],
                                                                      reg_type=args.reg_type, moving_avg=args.moving_avg,
                                                                      momentum=args.momentum_mvg,
                                                                      stat_type_list=args.stat_type,
                                                                      before_norm=args.before_norm))
                        break

    elif args.stat_reg == 'BNS':
        # todo regularization on BNS statistics
        # regularization on BNS staticstics
        bns_feature_hooks = []
        chosen_layers = choose_layers(model, candidate_bn_layers)
        for chosen_layer in chosen_layers:
            # regularize between manually computed target batch statistics (whether or not in running manner) between  source stastistics
            bns_feature_hooks.append(BNFeatureHook(chosen_layer, reg_type='l2norm', running_manner=args.running_manner,
                                                   use_src_stat_in_reg=args.use_src_stat_in_reg, momentum=args.momentum_bns))
    else:
        raise Exception(f'undefined regularization type {args.stat_reg}')

    epoch_result_list = []

    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips
    if args.if_sample_tta_aug_views:
        assert n_clips == 1
        n_augmented_views = args.n_augmented_views

    if_pred_consistency = args.if_pred_consistency if args.if_sample_tta_aug_views else False



    for epoch in range(args.n_epoch_adapat):
        batch_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_reg = AverageMeter()
        losses_consis = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        pred_concat = []
        gt_concat = []
        end = time.time()
        # with torch.autograd.set_detect_anomaly(True):
        for i, (input, target) in enumerate(tta_loader):  #
            # model.eval()
            model.train()  # BN layers are set to train mode
            if args.fix_BNS:  # fix the BNS during forward pass
                for m in model.modules():
                    for candidate in candidate_bn_layers:
                        if isinstance(m, candidate):
                            m.eval()
            actual_bz = input.shape[0]
            input = input.to(device)
            target = target.to(device)
            if args.arch == 'tanet':
                # (actual_bz,          C* spatial_crops * temporal_clips* clip_len,          256, 256) ->                  (actual_bz * spatial_crops * temporal_clips* clip_len,             C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                # input = input.view(actual_bz, -1, 3, input.size(2), input.size(3))
                # input = input.view(-1, 3, 224, 224)
                if args.if_sample_tta_aug_views:
                    input = input.view(actual_bz * args.test_crops * n_augmented_views,   args.clip_length,   3, input.size(2), input.size(3) )
                else:
                    input = input.view(actual_bz * args.test_crops * n_clips,
                                   args.clip_length, 3, input.size(2), input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,    C,     256, 256) -> (actual_bz * spatial_crops * temporal_clips,   clip_len,   C,  256,   256)
                output = model(input)  # (actual_bz * spatial_crops * temporal_clips,         clip_len,   C,   256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )

                # if_pred_consistency = False
                if args.if_sample_tta_aug_views:
                    output = output.reshape(actual_bz, args.test_crops * n_augmented_views, -1)  # (N, n_views, n_class )
                    if if_pred_consistency:
                        loss_consis = compute_pred_consis(output)
                    output = output.mean(1)
                else:
                    output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)

            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # todo (batch, n_views, C, T, H, W)
                if args.if_sample_tta_aug_views:
                    n_views = args.test_crops * n_augmented_views
                else:
                    n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                if args.if_sample_tta_aug_views:
                    if if_pred_consistency:
                        output, view_cls_score = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
                        loss_consis = compute_pred_consis(view_cls_score)
                else:
                    output, _ = model( input)
            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                # forward pass
                output = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = rearrange(output, '(d0 d1) d2 -> d0 d1 d2', d0=actual_bz)  # (batch * n_views,  n_class) ->  (batch, n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = torch.mean(output, dim=1)  # (batch, n_views,  n_class) ->  (batch,  n_class)
            loss_ce = criterion(output, target)

            loss_reg = torch.tensor(0).float().to(device)
            if args.stat_reg:
                for hook in stat_reg_hooks:
                    loss_reg += hook.r_feature.to(device)
            else:
                for hook in bns_feature_hooks:
                    loss_reg += hook.r_feature.to(device)
            if if_pred_consistency:
                loss = args.lambda_feature_reg*loss_reg + args.lambda_pred_consis * loss_consis
            else:
                loss = loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_iter += 1

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            _, preds = torch.max(output, 1)
            # pred_concat = np.concatenate([pred_concat, preds.detach().cpu().numpy()])
            # gt_concat = np.concatenate([gt_concat, target.detach().cpu().numpy()])

            losses_ce.update(loss_ce.item(), actual_bz)
            losses_reg.update(loss_reg.item(), actual_bz)
            if if_pred_consistency:
                losses_consis.update(loss_consis.item(), actual_bz)
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose:
                if i % args.print_freq == 0:
                    logger.debug(('TTA Epoch{epoch}: [{0}/{1}]\t'
                                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                  'Loss reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t'
                                  'Loss consis {loss_consis.val:.4f} ({loss_consis.avg:.4f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(tta_loader), epoch=epoch, batch_time=batch_time, loss_reg=losses_reg, loss_consis = losses_consis,
                        top1=top1, top5=top5)))

            if writer is not None:
                writer.add_scalars('loss', {'loss_reg': loss_reg.item()}, global_step=global_iter)
                if if_pred_consistency:
                    writer.add_scalars('loss', {'loss_consis': loss_consis.item()}, global_step=global_iter)
                writer.add_scalars('loss', {'loss_ce': loss_ce.item()}, global_step=global_iter)
            # writer.add_scalars('acc', {'val_acc': top1.avg}, global_step=epoch)
        # logger.debug(f'Validation acc {top1.avg} ')
        # logger.debug(classification_report(pred_concat, gt_concat))

        # evaluate on the entire test set at the end of each epoch
        # todo remove all the hooks
        if args.stat_reg in ['mean_var', 'cossim']:
            for stat_reg_hook in stat_reg_hooks:
                stat_reg_hook.close()
        elif args.stat_reg == 'BNS':
            for bns_feature_hook in bns_feature_hooks:
                bns_feature_hook.close()
        top1_acc = validate_brief(eval_loader=eval_loader, model=model, global_iter=global_iter, epoch=epoch, args=args,
                       logger=logger, writer=writer)
        epoch_result_list.append(top1_acc)
    return epoch_result_list, model



def evaluate_baselines(model, args=None, logger=None, writer=None):

    tta_loader = torch.utils.data.DataLoader(
        get_dataset(args, split='val'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    eval_loader = torch.utils.data.DataLoader(
        get_dataset(args, split='val'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    global_iter = 0
    validate_brief(eval_loader=eval_loader, model=model, global_iter=global_iter, args=args,
                   logger=logger, writer=writer)
