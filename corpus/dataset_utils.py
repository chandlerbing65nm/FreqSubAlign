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
# from corpus.model_utils import get_model
# from corpus.statistics_utils import compute_statistics, compute_cos_similarity, load_precomputed_statistics

def get_dataset(args, split='train'):
    train_augmentation = get_augmentation(args, args.modality, args.input_size)  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip

    train_transform = torchvision.transforms.Compose([
        train_augmentation,  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip
        fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
        GroupNormalize(args.input_mean, args.input_std)])

    dua_transform = torchvision.transforms.Compose([
        GroupScale(int(args.scale_size)),
        GroupCenterCrop(args.crop_size),
        fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
    ])

    val_transform = torchvision.transforms.Compose([
        GroupScale(int(args.scale_size)),
        GroupCenterCrop(args.crop_size),
        fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
        GroupNormalize(args.input_mean, args.input_std),
    ])
    if args.datatype == 'vid':
        if split == 'train':
            if args.tsn_style:  # vid, train, tsn
                return MyTSNVideoDataset(args, args.root_path, args.train_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         # load images
                                         transform=train_transform, video_data_dir=args.video_data_dir,
                                         debug=args.debug)
            else:  # vid, train,  non-tsn
                return MyVideoDataset(args, args.root_path, args.train_vid_list, clip_length=args.clip_length,
                                      frame_interval=args.frame_interval, num_clips=args.num_clips,
                                      modality=args.modality, vid_format=args.vid_format,
                                      transform=train_transform, video_data_dir=args.video_data_dir, debug=args.debug)


        use_dua_val = False
        if args.evaluate_baselines:
            if args.baseline == 'dua':
                use_dua_val = True  #  use dua transformation for evaluation

        if split == 'val' and (not use_dua_val):
            if args.tsn_style:  # vid, val, tsn
                return MyTSNVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         transform=val_transform, video_data_dir=args.video_data_dir, test_mode=True,
                                         debug=args.debug)
            else:  # vid, val, non-tsn
                return MyVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                      frame_interval=args.frame_interval, num_clips=args.num_clips,
                                      modality=args.modality, vid_format=args.vid_format,
                                      transform=val_transform, test_mode=True, video_data_dir=args.video_data_dir,
                                      debug=args.debug)

        elif split == 'val' and use_dua_val:
            if args.tsn_style:  # vid, val, tsn
                return MyTSNVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         transform=dua_transform, video_data_dir=args.video_data_dir, test_mode=True,
                                         debug=args.debug), \
                       MyTSNVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                         frame_interval=args.frame_interval,
                                         num_clips=args.num_clips, modality=args.modality, vid_format=args.vid_format,
                                         transform=val_transform, video_data_dir=args.video_data_dir, test_mode=True,
                                         debug=args.debug)
            else:  # vid, val, non-tsn
                return MyVideoDataset(args, args.root_path, args.val_vid_list, clip_length=args.clip_length,
                                      frame_interval=args.frame_interval, num_clips=args.num_clips,
                                      modality=args.modality, vid_format=args.vid_format,
                                      transform=val_transform, test_mode=True, video_data_dir=args.video_data_dir,
                                      debug=args.debug)

    elif args.datatype == 'frame':
        if split == 'train':
            if args.tsn_style:  # frame, train, tsn
                return MyTSNDataset(args, args.root_path, args.train_frame_list, clip_length=args.clip_length,
                                    frame_interval=args.frame_interval,
                                    num_clips=args.num_clips, modality=args.modality,
                                    image_tmpl=args.img_tmpl if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
                                    # load images
                                    transform=train_transform, data_dir=args.frame_data_dir, debug=args.debug)
            else:  # vid, train, non-tsn
                raise Exception('not implemented yet!')
        elif split == 'val':
            if args.tsn_style:  # frame, val, tsn
                return MyTSNDataset(args, args.root_path, args.val_frame_list, clip_length=args.clip_length,
                                    frame_interval=args.frame_interval,
                                    num_clips=args.num_clips, modality=args.modality,
                                    image_tmpl=args.img_tmpl if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
                                    transform=val_transform, data_dir=args.frame_data_dir, test_mode=True,
                                    debug=args.debug)
            else:  # vid, val, non-tsn
                raise Exception('not implemented yet!')



def get_dataset_tanet(args, split = 'train', dataset_type = None):
    from models.tanet_models.transforms import GroupFullResSample_TANet, GroupScale_TANet, GroupCenterCrop_TANet, Stack_TANet, ToTorchFormatTensor_TANet, \
        GroupNormalize_TANet, GroupMultiScaleCrop_TANet, SubgroupWise_MultiScaleCrop_TANet, SubgroupWise_RandomHorizontalFlip_TANet
    from models.tanet_models.video_dataset import Video_TANetDataSet
    
    if split == 'train':
        raise NotImplementedError('Training dataset processing for TANet to be added!')
    elif split == 'val':
        # if args.full_res,  feed 256x256 to the network
        # input_size = tanet_model.scale_size if args.full_res else tanet_model.input_size
        input_size = args.scale_size if args.full_res else args.input_size

        if dataset_type == 'tta':
            if_sample_tta_aug_views = args.if_sample_tta_aug_views
        elif dataset_type == 'eval':
            if_sample_tta_aug_views = False

        tta_view_sample_style_list = args.tta_view_sample_style_list if if_sample_tta_aug_views else None
        n_augmented_views = args.n_augmented_views if if_sample_tta_aug_views else None
        if_spatial_rand_cropping = args.if_spatial_rand_cropping if if_sample_tta_aug_views else False

        if args.test_crops == 1:
            if if_spatial_rand_cropping:
                # cropping = torchvision.transforms.Compose([GroupMultiScaleCrop_TANet(input_size)] )
                cropping = torchvision.transforms.Compose([SubgroupWise_MultiScaleCrop_TANet(input_size=input_size,
                                                                                             n_temp_clips=n_augmented_views,
                                                                                             clip_len=args.clip_length)])
                # label_transforms =  {
                # 86: 87,
                # 87: 86,
                # 93: 94,
                # 94: 93,
                # 166: 167,
                # 167: 166 } if args.dataset == 'somethingv2' else None
                # cropping = torchvision.transforms.Compose([SubgroupWise_MultiScaleCrop_TANet(input_size=input_size,
                #                                                                              n_temp_clips=n_augmented_views,
                #                                                                              clip_len=args.clip_length),
                #                                            SubgroupWise_RandomHorizontalFlip_TANet(label_transforms= label_transforms,
                #                                                                                    n_temp_clips=n_augmented_views,
                #                                                                                    clip_len=args.clip_length)])
            else:
                cropping = torchvision.transforms.Compose([  # scale to scale_size, then center crop to input_size
                    GroupScale_TANet(args.scale_size), # scale size is 256, input_size is 224, todo here scale_size is the size of the smaller edge
                    GroupCenterCrop_TANet(input_size),
                ])
        elif args.test_crops == 3:
            cropping = torchvision.transforms.Compose([GroupFullResSample_TANet(input_size, args.scale_size, flip=False)])
        else:
            raise NotImplementedError(f'{args.test_crops} spatial crops not implemented!')



        return Video_TANetDataSet(
            args.val_vid_list,
            num_segments=args.clip_length,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            # image_tmpl=prefix,
            vid_format=args.vid_format,
            test_mode=True,
            remove_missing= True,
            transform=torchvision.transforms.Compose([
                cropping,   #  GroupFullResSample,  scale to scale size, and crop  left, right, center  3 spatial crops    10 temporal clips,  16 frames,     480 frames in total   256 256
                Stack_TANet(roll=False),  #  stack the temporal dimension into channel dimension,   (*, C, T, H, W) -> (*, C*T, H, W)   ( *,   480*3, 256, 256)
                ToTorchFormatTensor_TANet(div= True),  # todo divide by 255   [0.485, 0.456, 0.406] * 255 = [123.675, 116.28, 103.53]      [0.229, 0.224, 0.225] * 255= [58.395,57.12, 57.375]
                GroupNormalize_TANet(args.input_mean, args.input_std),
            ]),
            video_data_dir=args.video_data_dir,
            test_sample=args.sample_style, #  'uniform-x' or 'dense-x'
            debug=args.debug,
            if_sample_tta_aug_views= if_sample_tta_aug_views,
            tta_view_sample_style_list=tta_view_sample_style_list,
            n_tta_aug_views=n_augmented_views)




def get_dataset_tanet_dua(args, tanet_model = None,  split = 'train'):
    from models.tanet_models.transforms import GroupFullResSample_TANet, GroupScale_TANet, GroupCenterCrop_TANet, Stack_TANet, ToTorchFormatTensor_TANet, GroupNormalize_TANet
    from models.tanet_models.video_dataset import Video_TANetDataSet
    if split == 'train':
        raise NotImplementedError('Training dataset processing for TANet to be added!')
    elif split == 'val':
        # if args.full_res,  feed 256x256 to the network
        input_size = tanet_model.scale_size if args.full_res else tanet_model.input_size
        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose([  # scale to scale_size, then center crop to input_size
                GroupScale_TANet(tanet_model.scale_size),
                GroupCenterCrop_TANet(input_size),
            ])
        elif args.test_crops == 3:
            cropping = torchvision.transforms.Compose([GroupFullResSample_TANet(input_size, tanet_model.scale_size, flip=False)])
        else:
            raise NotImplementedError(f'{args.test_crops} spatial crops not implemented!')

        return Video_TANetDataSet(
            args.val_vid_list,
            num_segments=args.clip_length,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            # image_tmpl=prefix,
            vid_format=args.vid_format,
            test_mode=True,
            remove_missing= True,
            transform=torchvision.transforms.Compose([
                cropping,   #  GroupFullResSample,  scale to scale size, and crop  left, right, center  3 spatial crops    10 temporal clips,  16 frames,     480 frames in total   256 256
                Stack_TANet(roll=False),  #  stack the temporal dimension into channel dimension,   (*, C, T, H, W) -> (*, C*T, H, W)   ( *,   480*3, 256, 256)
                ToTorchFormatTensor_TANet(div= True), # todo divide by 255
                GroupNormalize_TANet(tanet_model.input_mean, tanet_model.input_std),
            ]), video_data_dir=args.video_data_dir,
            test_sample=args.sample_style,
            debug=args.debug),\
            Video_TANetDataSet( args.val_vid_list,
            num_segments=args.clip_length,
            new_length=1 if args.modality == "RGB" else 5,
            modality=args.modality,
            # image_tmpl=prefix,
            vid_format=args.vid_format,
            test_mode=True,
            remove_missing=True,
            transform=torchvision.transforms.Compose([
                # cropping,
                # GroupFullResSample,  scale to scale size, and crop  left, right, center  3 spatial crops    10 temporal clips,  16 frames,     480 frames in total   256 256
                Stack_TANet(roll=False),
                # stack the temporal dimension into channel dimension,   (*, C, T, H, W) -> (*, C*T, H, W)   ( *,   480*3, 256, 256)
                ToTorchFormatTensor_TANet(div=True),
                # GroupNormalize_TANet(tanet_model.input_mean, tanet_model.input_std),
            ]),
            video_data_dir=args.video_data_dir,
            test_sample=args.sample_style,
            debug=args.debug)



def get_dataset_videoswin(args, split = 'train', dataset_type = None):
    from models.videoswintransformer_models.video_dataset import Video_SwinDataset
    # from models.videoswintransformer_models.trans
    if split == 'train':
        raise NotImplementedError('Training dataset processing for Video Swin Transformer to be added!')
    elif split == 'val':
        if dataset_type == 'tta':
            if_sample_tta_aug_views = args.if_sample_tta_aug_views
        elif dataset_type == 'eval':
            if_sample_tta_aug_views = False
        tta_view_sample_style_list = args.tta_view_sample_style_list if if_sample_tta_aug_views else None
        return Video_SwinDataset( args.val_vid_list,
                 num_segments=args.clip_length,  # clip_length
                 frame_interval = args.frame_interval,
                 num_clips = args.num_clips, # number of temporal clips
                 frame_uniform = args.frame_uniform,
                 test_mode = True,
                 flip_ratio = args.flip_ratio,
                 scale_size = args.scale_size,
                 input_size= args.input_size,
                 img_norm_cfg  = args.img_norm_cfg,

                 vid_format=args.vid_format,
                 video_data_dir = args.video_data_dir,
                 remove_missing = False,
                 debug = args.debug,
                if_sample_tta_aug_views= if_sample_tta_aug_views,
                tta_view_sample_style_list=tta_view_sample_style_list,
                n_augmented_views=args.n_augmented_views )
