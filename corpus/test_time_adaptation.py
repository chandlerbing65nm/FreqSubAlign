import time
# import os
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
import torch.nn.functional as F
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
from utils.ema_teacher import EMATeacher


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
from torch.nn.parallel import DistributedDataParallel

# from corpus.training import train, validate, validate_brief
from corpus.dataset_utils import get_dataset, get_dataset_tanet, get_dataset_videoswin
# from corpus.model_utils import get_model
from corpus.statistics_utils import compute_statistics, compute_cos_similarity, load_precomputed_statistics
from utils.confidence_gated_optimizer import ConfidenceGatedOptimizer, AdaptiveConfidenceGatedOptimizer
from utils.confidence_meta_optimizer import ConfidenceMetaOptimizer, AdaptiveConfidenceMetaOptimizer
from utils.confidence_inverted_optimizer import ConfidenceInvertedOptimizer, AdaptiveConfidenceInvertedOptimizer

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
        # Multi-epoch support: removed hard-coded n_epoch_adapat == 1 assertion
        logger.debug(f"TTA Standard mode: Using {args.n_epoch_adapat} epochs for adaptation")
    elif args.if_tta_standard == 'tta_online':
        assert args.momentum_mvg != 1.0  # todo accumulate the target statistics for different samples
        assert args.n_gradient_steps == 1 # todo one gradient step per sample (on forward pass per sample )
        # Multi-epoch support: removed hard-coded n_epoch_adapat == 1 assertion
        logger.debug(f"TTA Online mode: Using {args.n_epoch_adapat} epochs for adaptation")
    from utils.norm_stats_utils import CombineNormStatsRegHook_onereg
    # from utils.relation_map_utils import CombineCossimRegHook

    # candidate_bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    candidate_bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]


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
    losses_distill = AverageMeter()
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
            # bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
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
        # Multi-epoch adaptation: Store original input/target for multiple epochs
        original_input, original_target = input.clone(), target.clone()
        
        # Multi-epoch training loop
        for epoch_id in range(args.n_epoch_adapat):
            # print(f"Batch {batch_id}, Epoch {epoch_id+1}/{args.n_epoch_adapat}")
            
            # Use original input/target for each epoch
            input, target = original_input.clone(), original_target.clone()
            
            setup_model_optimizer = False
            if args.if_tta_standard == 'tta_standard':
                # For multi-epoch standard TTA, setup model/optimizer only at the beginning of first epoch
                if epoch_id == 0:
                    setup_model_optimizer = True  #  setup model and optimizer before adaptation starts
            elif args.if_tta_standard == 'tta_online':
                if batch_id == 0 and epoch_id == 0:
                    setup_model_optimizer = True #  setup model and optimizer only before the first sample comes

            if setup_model_optimizer:
                # print(f'Batch {batch_id}, Epoch {epoch_id+1}, initialize the model, update chosen layers, initialize hooks, intialize average meter')
                # todo ############################################################
                # todo #####################################  re-intialize the model, update chosen_layers from this new model
                # todo ############################################################
                model = cp.deepcopy(model_origin)
                
                # Initialize EMA teacher if enabled
                ema_teacher = None
                if hasattr(args, 'use_ema_teacher') and args.use_ema_teacher:
                    from utils.ema_teacher import EMATeacher
                    ema_teacher = EMATeacher(
                        model=model,
                        momentum=args.ema_momentum,
                        temperature=args.ema_temperature,
                        adaptive_temperature=args.ema_adaptive_temp,
                        min_temperature=args.ema_min_temp,
                        max_temperature=args.ema_max_temp,
                        temperature_alpha=args.ema_temp_alpha,
                        device=device
                    )
                    logger.info(f"EMA teacher initialized with momentum={args.ema_momentum}")
                # when we initialize the model, we have to re-choose the layers from it.
                if args.arch == 'tanet':
                    # todo  temporal statistics are computed on  nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                    #      spatial statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,   not on Batchnorm1d
                    #      spatiotemporal statistics are computed on nn.BatchNorm2d, nn.BatchNorm3d,  not on Batchnorm1d
                    # bn_layers = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
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
                    base_optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
                else:
                    base_optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                                weight_decay=args.weight_decay)
                
                # Create optimizer with CIMO/CGMO/CGO wrapping
                if hasattr(args, 'use_cimo') and args.use_cimo:
                    # Use Confidence-Inverted Meta-Optimizer (CIMO)
                    if hasattr(args, 'cimo_adaptive') and args.cimo_adaptive:
                        optimizer = AdaptiveConfidenceInvertedOptimizer(
                            base_optimizer,
                            confidence_threshold=args.cimo_confidence_threshold,
                            min_lr_scale=args.cimo_min_lr_scale,
                            max_lr_scale=args.cimo_max_lr_scale,
                            confidence_power=args.cimo_confidence_power,
                            enable_momentum_correction=args.cimo_enable_momentum_correction,
                            enable_logging=True
                        )
                    else:
                        optimizer = ConfidenceInvertedOptimizer(
                            base_optimizer,
                            confidence_threshold=args.cimo_confidence_threshold,
                            min_lr_scale=args.cimo_min_lr_scale,
                            max_lr_scale=args.cimo_max_lr_scale,
                            confidence_power=args.cimo_confidence_power,
                            enable_momentum_correction=args.cimo_enable_momentum_correction,
                            enable_logging=True
                        )
                elif hasattr(args, 'use_cgmo') and args.use_cgmo:
                    # Use CGMO (Meta-Optimizer) - prioritizes over CGO
                    if hasattr(args, 'cgmo_adaptive_threshold') and args.cgmo_adaptive_threshold:
                        optimizer = AdaptiveConfidenceMetaOptimizer(
                            base_optimizer,
                            confidence_threshold=args.cgmo_confidence_threshold,
                            min_lr_scale=args.cgmo_min_lr_scale,
                            max_lr_scale=args.cgmo_max_lr_scale,
                            confidence_power=args.cgmo_confidence_power,
                            enable_momentum_correction=args.cgmo_enable_momentum_correction,
                            target_confidence=args.cgmo_confidence_threshold,
                            enable_logging=args.cgo_enable_logging
                        )
                        logger.info(f"Using Adaptive CGMO with threshold={args.cgmo_confidence_threshold}")
                    else:
                        optimizer = ConfidenceMetaOptimizer(
                            base_optimizer,
                            confidence_threshold=args.cgmo_confidence_threshold,
                            min_lr_scale=args.cgmo_min_lr_scale,
                            max_lr_scale=args.cgmo_max_lr_scale,
                            confidence_power=args.cgmo_confidence_power,
                            enable_momentum_correction=args.cgmo_enable_momentum_correction,
                            enable_logging=args.cgo_enable_logging
                        )
                        logger.info(f"Using CGMO with threshold={args.cgmo_confidence_threshold}")
                elif hasattr(args, 'use_cgo') and args.use_cgo:
                    # Fallback to original CGO
                    if hasattr(args, 'cgo_adaptive') and args.cgo_adaptive:
                        optimizer = AdaptiveConfidenceGatedOptimizer(
                            base_optimizer,
                            initial_threshold=args.cgo_confidence_threshold,
                            target_adaptation_rate=args.cgo_target_adaptation_rate,
                            confidence_metric=args.cgo_confidence_metric,
                            enable_logging=args.cgo_enable_logging
                        )
                        logger.info(f"Using Adaptive CGO with initial threshold={args.cgo_confidence_threshold}")
                    else:
                        optimizer = ConfidenceGatedOptimizer(
                            base_optimizer,
                            confidence_threshold=args.cgo_confidence_threshold,
                            confidence_metric=args.cgo_confidence_metric,
                            enable_logging=args.cgo_enable_logging
                        )
                        logger.info(f"Using CGO with threshold={args.cgo_confidence_threshold}")
                else:
                    optimizer = base_optimizer
                    logger.info("Using standard optimizer (no CGMO/CGO)")

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
                    raw_output = model(input)
                    if args.if_sample_tta_aug_views:
                        raw_output = raw_output.reshape(actual_bz, args.test_crops * n_augmented_views, -1)  # (N, n_views, n_class )
                        if if_pred_consistency:
                            loss_consis = compute_pred_consis(raw_output)
                        output = raw_output.mean(1)
                    else:
                        output = raw_output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)

                elif args.arch == 'videoswintransformer':
                    if args.if_sample_tta_aug_views:
                        n_views = args.test_crops * n_augmented_views
                    else:
                        n_views = args.test_crops * n_clips
                    # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                    if args.if_sample_tta_aug_views:
                        if if_pred_consistency:
                            raw_output, view_cls_score = model( input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
                            loss_consis = compute_pred_consis(view_cls_score)
                    else:
                        raw_output, _ = model( input)
                    output = raw_output
                else:
                    raise NotImplementedError(f'Incorrect model type {args.arch}')

                loss_ce = criterion(output, target)
                
                loss_reg = torch.tensor(0).float().to(device)
                if args.stat_reg:
                    for hook in stat_reg_hooks:
                        loss_reg += hook.r_feature.to(device)
                else:
                    raise Exception(f'undefined regularization type {args.stat_reg}')
                
                # Compute total loss
                loss_components = []
                
                # Always include feature regularization
                loss_components.append(args.lambda_feature_reg * loss_reg)
                
                # Include prediction consistency if enabled
                if if_pred_consistency:
                    loss_components.append(args.lambda_pred_consis * loss_consis)
                    
                    # Conditionally include CE loss when using consistency
                    if hasattr(args, 'include_ce_in_consistency') and args.include_ce_in_consistency:
                        loss_components.append(loss_ce)
                # else:
                #     # Always include CE loss when no consistency
                #     loss_components.append(loss_ce)
                
                # Add distillation loss from EMA teacher
                loss_distill = torch.tensor(0.0).float().to(device)
                if hasattr(args, 'use_ema_teacher') and args.use_ema_teacher and ema_teacher is not None:
                    # Compute confidence for inverted weighting
                    confidence = torch.max(F.softmax(output, dim=-1), dim=-1)[0]
                    
                    # Warmup: optionally skip distillation for first K steps
                    if getattr(args, 'ema_distill_warmup_steps', 0) and step_id < args.ema_distill_warmup_steps:
                        pass
                    else:
                        # Define n_views based on architecture and augmentation settings
                        if args.arch == 'tanet':
                            if args.if_sample_tta_aug_views:
                                n_views = args.test_crops * n_augmented_views
                            else:
                                n_views = args.test_crops * n_clips
                        elif args.arch == 'videoswintransformer':
                            if args.if_sample_tta_aug_views:
                                n_views = args.test_crops * n_augmented_views
                            else:
                                n_views = args.test_crops * n_clips
                        else:
                            n_views = args.test_crops * n_clips  # Default fallback
                        
                        # Get teacher predictions (multi-view aggregation)
                        if args.if_sample_tta_aug_views:
                            # For multi-view case, aggregate teacher predictions across views
                            teacher_logits_list = []
                            for v in range(n_views):
                                start_idx = v * actual_bz
                                end_idx = (v + 1) * actual_bz
                                view_input = input[start_idx:end_idx] if args.arch == 'tanet' else input
                                teacher_logits = ema_teacher.forward(view_input, confidence)
                                teacher_logits_list.append(teacher_logits)
                            
                            # Aggregate teacher predictions
                            teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
                        else:
                            # Single view case
                            teacher_logits = ema_teacher.forward(input, confidence)
                        
                        # Compute distillation loss with a single, shared temperature
                        T = args.ema_temperature
                        student_logits_T = output / T
                        teacher_targets = F.softmax(teacher_logits / T, dim=-1)
                        # Per-sample KL for weighting
                        kl_per_sample = F.kl_div(
                            F.log_softmax(student_logits_T, dim=-1),
                            teacher_targets,
                            reduction='none'
                        ).sum(dim=-1)
                        
                        # Confidence-inverted sample weighting: higher weight for low-confidence samples
                        p = getattr(args, 'distill_conf_power', 1.5)
                        weight = (1.0 - confidence).pow(p)
                        # Optional gating by confidence threshold
                        conf_thresh = getattr(args, 'ema_distill_conf_thresh', 1.0)
                        if conf_thresh < 1.0:
                            gate = (confidence < conf_thresh).float()
                            weight = weight * gate
                        weight = torch.clamp(weight, 0.2, 1.0)
                        
                        # If all weights are zero (fully gated), skip contribution
                        denom = weight.sum().clamp_min(1e-6)
                        loss_distill = (kl_per_sample * weight).sum() / denom
                        
                        loss_components.append(args.lambda_distill * loss_distill)
                
                loss = sum(loss_components)

                optimizer.zero_grad()
                loss.backward()
                
                # Use CIMO/CGMO/CGO conditional step if enabled, otherwise standard step
                if hasattr(args, 'use_cimo') and args.use_cimo:
                    # Use CIMO (Inverted Meta-Optimizer) - confidence-inverted updates
                    step_info = optimizer.conditional_step(raw_output, loss)
                    # Log CIMO decision if verbose
                    if args.verbose and step_info:
                        logger.debug(f"CIMO Step - Confidence: {step_info['confidence']:.3f}, "
                                   f"LR Scale: {step_info['lr_scale']:.3f} (INVERTED), "
                                   f"Gradient Weight: {step_info['gradient_weight']:.3f}")
                elif hasattr(args, 'use_cgmo') and args.use_cgmo:
                    # Use CGMO (Meta-Optimizer) - confidence-weighted updates
                    step_info = optimizer.conditional_step(raw_output, loss)
                    # Log CGMO decision if verbose
                    if args.verbose and step_info:
                        logger.debug(f"CGMO Step - Confidence: {step_info['confidence']:.3f}, "
                                   f"LR Scale: {step_info['lr_scale']:.3f}, "
                                   f"Gradient Weight: {step_info['gradient_weight']:.3f}")
                elif hasattr(args, 'use_cgo') and args.use_cgo:
                    # Use CGO - binary gating
                    step_info = optimizer.conditional_step(raw_output, loss)
                    # Log CGO decision if verbose
                    if args.verbose and step_info:
                        logger.debug(f"CGO Step - Confidence: {step_info['confidence']:.3f}, "
                                   f"Threshold: {step_info['threshold']:.3f}, "
                                   f"Adapted: {step_info['adapted']}")
                else:
                    optimizer.step()

                # Update EMA teacher after optimizer step
                if hasattr(args, 'use_ema_teacher') and args.use_ema_teacher and ema_teacher is not None:
                    ema_teacher.update(model)

                losses_ce.update(loss_ce.item(), actual_bz)
                losses_reg.update(loss_reg.item(), actual_bz)
                if if_pred_consistency:
                    losses_consis.update(loss_consis.item(), actual_bz)
                if hasattr(args, 'use_ema_teacher') and args.use_ema_teacher and ema_teacher is not None:
                    losses_distill.update(loss_distill.item(), actual_bz)

            # todo ############################################################
            # todo ##################################### remove all the hooks, no computation of statistics during inference
            # todo ############################################################
            if args.stat_reg:
                for stat_reg_hook in stat_reg_hooks:
                    stat_reg_hook.close()
            else:
                raise Exception(f'undefined regularization type {args.stat_reg}')

            # todo ##########################################################################################
            # todo ################### Inference on evaluation batch (only for last epoch) ##############################################
            # todo ##########################################################################################
            # Only evaluate on the last epoch to avoid exhausting the eval_loader_iterator
            if epoch_id == args.n_epoch_adapat - 1:
                model.eval()
                with torch.no_grad():
                    try:
                        eval_input, eval_target = next(eval_loader_iterator)
                    except StopIteration:
                        # If eval_loader is exhausted, reinitialize it
                        eval_loader_iterator = iter(eval_loader)
                        eval_input, eval_target = next(eval_loader_iterator)
                    
                    eval_input, eval_target = eval_input.to(device), eval_target.to(device)
                    actual_eval_bz = eval_input.shape[0]
                    
                    if args.arch == 'tanet':
                        # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                        eval_input = eval_input.view(-1, 3, eval_input.size(2), eval_input.size(3))
                        eval_input = eval_input.view(actual_eval_bz * args.test_crops * n_clips,
                                           args.clip_length, 3, eval_input.size(2), eval_input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                        output = model(eval_input)  # (actual_bz * spatial_crops * temporal_clips, clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                        # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )
                        output = output.reshape(actual_eval_bz, args.test_crops * n_clips, -1).mean(1)
                    elif args.arch == 'videoswintransformer':
                        # the format shape is N C T H W         if  collapse in datsaet is True, then shape is  (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)
                        # (batch, n_views, C, T, H, W)
                        n_views = args.test_crops * n_clips
                        # eval_input = eval_input.view(-1, n_views, 3, args.clip_length, eval_input.size(3), eval_input.size(4))
                        output, _ = model(eval_input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
                    else:
                        raise NotImplementedError(f'Incorrect model type {args.arch}')
                    prec1, prec5 = accuracy(output.data, eval_target, topk=(1, 5))
                    top1.update(prec1.item(), actual_eval_bz)
                    top5.update(prec5.item(), actual_eval_bz)

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
                              'Loss distill {loss_distill.val:.4f} ({loss_distill.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_id, len(tta_loader), epoch=epoch_id+1, batch_time=batch_time, loss_reg=losses_reg, loss_consis=losses_consis,
                    loss_distill=losses_distill, top1=top1, top5=top5)))

        # # Log CIMO/CGMO/CGO statistics if applicable
        # if hasattr(optimizer, 'get_adaptation_stats') and (hasattr(args, 'use_cimo') and args.use_cimo):
        #     stats = optimizer.get_adaptation_stats()
        #     if logger:
        #         logger.info(f"CIMO Stats - Mean Confidence: {stats['mean_confidence']:.3f}, "
        #                    f"Mean LR Scale: {stats['mean_lr_scale']:.3f}, "
        #                    f"Adaptation Rate: {stats['adaptation_rate']:.3f} (INVERTED)")
        # elif hasattr(optimizer, 'get_adaptation_stats') and (hasattr(args, 'use_cgmo') and args.use_cgmo):
        #     stats = optimizer.get_adaptation_stats()
        #     if logger:
        #         logger.info(f"CGMO Stats - Mean Confidence: {stats['mean_confidence']:.3f}, "
        #                    f"Mean LR Scale: {stats['mean_lr_scale']:.3f}, "
        #                    f"Adaptation Rate: {stats['adaptation_rate']:.3f}")
        # elif hasattr(optimizer, 'get_adaptation_stats') and (hasattr(args, 'use_cgo') and args.use_cgo):
        #     stats = optimizer.get_adaptation_stats()
        #     if logger:
        #         logger.info(f"CGO Stats - Mean Confidence: {stats['mean_confidence']:.3f}, "
        #                    f"Adaptation Rate: {stats['adaptation_rate']:.3f}, "
        #                    f"Gated Updates: {stats['gated_updates']}/{stats['total_samples']}")

    epoch_result_list.append(top1.avg)
    
    # model_path = osp.join(  args.result_dir, f'{args.corruptions}.model' )
    # logger.debug(f'Saving models to {model_path}')
    #
    # torch.save( model.state_dict(), model_path )

    return epoch_result_list


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