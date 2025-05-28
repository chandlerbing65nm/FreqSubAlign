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
# from corpus.dataset_utils import get_dataset, get_dataset_tanet, get_dataset_videoswin
# from corpus.statistics_utils import compute_statistics, compute_cos_similarity, load_precomputed_statistics

def get_model(args, num_classes, logger):
    if args.arch == 'r2plus1d':
        model = MyR2plus1d(num_classes=num_classes, use_pretrained=args.use_pretrained)
    elif args.arch == 'i3d_incep':
        model = InceptionI3d(num_classes=400, in_channels=3)
        if args.use_pretrained:
            model.load_state_dict(torch.load(args.pretrained_model))
            logger.debug(f'Loaded pretrained I3D Inception model {args.pretrained_model}')
        model.replace_logits(num_classes=num_classes)
    elif 'i3d_resnet' in args.arch:
        if args.arch in ['i3d_resnet18', 'i3d_resnet34', ]:
            in_channel = 512
        elif args.arch in ['i3d_resnet50', 'i3d_resnet101', 'i3d_resnet152', ]:
            in_channel = 2048
        model = getattr(i3d, args.arch)(modality=args.modality, num_classes=num_classes, in_channel=in_channel,
                                        dropout_ratio=args.dropout)
    elif args.arch == 'tanet':
        model = TSN(
            num_classes,
            args.clip_length,
            args.modality,
            base_model='resnet50',
            consensus_type= 'avg',
            img_feature_dim=args.img_feature_dim,
            tam= True,
            non_local=False,
            partial_bn=args.partial_bn
        )
    elif args.arch == 'videomae':
        model = create_model(
            args.model, # 'vit_base_patch16_224'
            pretrained=False,
            num_classes= num_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )
    elif args.arch == 'videoswintransformer':
        model = Recognizer3D(num_classes =  num_classes, patch_size= args.patch_size, window_size=args.window_size, drop_path_rate=args.drop_path_rate)
    else:
        raise Exception(f'{args.arch} is not a valid model!')
    return model
