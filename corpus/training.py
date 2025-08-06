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

# from corpus.test_time_adaptation import tta_standard, test_time_adapt, evaluate_baselines
# from corpus.dataset_utils import get_dataset, get_dataset_tanet, get_dataset_videoswin
# from corpus.model_utils import get_model
# from corpus.statistics_utils import compute_statistics, compute_cos_similarity, load_precomputed_statistics

def train(train_loader, model, criterion, optimizer, epoch, args=None, logger=None, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):  # input (bz, n_clips,  3, clip_len, 224, 224), target (bz, )
        input = input.reshape(
            (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
        target = target.reshape((target.shape[0], 1)).repeat(1, args.num_clips)
        target = target.reshape((-1,) + target.shape[2:])  # (batch * n_views, )

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose:
            if i % args.print_freq == 0:
                logger.debug(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    writer.add_scalars('loss', {'losses': losses.avg}, global_step=epoch)
    writer.add_scalars('acc', {'train_acc': top1.avg}, global_step=epoch)
    writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], global_step=epoch)


def validate(val_loader, model, criterion, iter, epoch=None, args=None, logger=None, writer=None, optimizer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode

    pred_concat = []
    gt_concat = []
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips

    if args.evaluate_baselines:
        if args.baseline == 'source':
            logger.debug(f'Starting ---- {args.corruptions} ---- evaluation for Source...')
        elif args.baseline == 'tent':
            from baselines.tent import forward_and_adapt
            logger.debug(f'Starting ---- {args.corruptions} ---- adaptation for TENT...')
            for i, (input, target) in enumerate(val_loader):
                actual_bz = input.shape[0]
                input = input.to(device)
                if args.arch == 'tanet':
                    input = input.view(-1, 3, input.size(2), input.size(3))
                    input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2), input.size(3))
                    _ = forward_and_adapt(input, model, optimizer, args, actual_bz, n_clips)
                else:
                    input = input.reshape((-1,) + input.shape[2:])
                    _ = forward_and_adapt(input, model, optimizer)
            logger.debug(f'TENT Adaptation Finished --- Now Evaluating')
        elif args.baseline == 'norm':
            logger.debug(f'Starting ---- {args.corruptions} ---- adaptation for NORM...')
            with torch.no_grad():
                for i, (input, target) in enumerate(val_loader):
                    actual_bz = input.shape[0]
                    input = input.to(device)
                    if args.arch == 'tanet':
                        input = input.view(-1, 3, input.size(2), input.size(3))
                        input = input.view(actual_bz * args.test_crops * n_clips,
                                           args.clip_length, 3, input.size(2), input.size(3))
                        _ = model(input)
                    else:
                        input = input.reshape((-1,) + input.shape[2:])
                        _ = model(input)
            logger.debug(f'NORM Adaptation Finished --- Now Evaluating')
        elif args.baseline == 'shot':
            logger.debug(f'Starting ---- {args.corruptions} ---- evaluation for SHOT...')
        elif args.baseline == 'dua':
            logger.debug(f'Starting ---- {args.corruptions} ---- evaluation for DUA...')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):  #
            actual_bz = input.shape[0]
            input = input.to(device)
            target = target.to(device)
            if args.arch == 'tanet':
                # (actual_bz,    C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2),input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)

                output = model(input) #  (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W
                # (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)   -> (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                output, _ = model(  input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores



            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                output = model( input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction

                output = torch.squeeze(output)
                output = rearrange(output, '(d0 d1) d2 -> d0 d1 d2', d0=actual_bz)  # (batch * n_views,  n_class) ->  (batch, n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = torch.mean(output, dim=1)  # (batch, n_views,  n_class) ->  (batch,  n_class), take the average scores of multiple views

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            _, preds = torch.max(output, 1)
            # pred_concat = np.concatenate([pred_concat, preds.detach().cpu().numpy()])
            # gt_concat = np.concatenate([gt_concat, target.detach().cpu().numpy()])

            losses.update(loss.item(), actual_bz)
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose:
                if i % args.print_freq == 0:
                    logger.debug(('Test: [{0}/{1}]\t'
                                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

    logger.debug(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1,
                                                                                                            top5=top5,
                                                                                                            loss=losses)))
    if writer is not None:
        writer.add_scalars('loss', {'val_loss': losses.avg}, global_step=epoch)
        writer.add_scalars('acc', {'val_acc': top1.avg}, global_step=epoch)
    logger.debug(f'Validation acc {top1.avg} ')

    # logger.debug(classification_report(pred_concat, gt_concat))

    return top1.avg



def validate_brief(eval_loader, model, global_iter, epoch=None, args=None, logger=None, writer=None):
    batch_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    if args.arch == 'tanet':
        n_clips = int(args.sample_style.split("-")[-1])
    elif args.arch == 'videoswintransformer':
        n_clips = args.num_clips

    pred_concat = []
    gt_concat = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(eval_loader):  #
            model.eval()
            actual_bz = input.shape[0]
            input = input.to(device)
            target = target.to(device)
            if args.arch == 'tanet':
                # (actual_bz, C* spatial_crops * temporal_clips* clip_len, 256, 256) ->   (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256)
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2),input.size(3))  # (actual_bz * spatial_crops * temporal_clips* clip_len,  C, 256, 256) -> (actual_bz * spatial_crops * temporal_clips,  clip_len,  C, 256, 256)
                output = model(input) #  (actual_bz * spatial_crops * temporal_clips,         clip_len,  C, 256, 256)   ->     (actual_bz * spatial_crops * temporal_clips,       n_class )
                # take the average among all spatial_crops * temporal_clips,   (actual_bz * spatial_crops * temporal_clips,       n_class )  ->   (actual_bz,       n_class )
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
            elif args.arch == 'videoswintransformer':
                # the format shape is N C T H W         if  collapse in datsaet is True, then shape is  (actual_bz,   C* spatial_crops * temporal_clips* clip_len,    256,     256)
                # (batch, n_views, C, T, H, W)
                n_views = args.test_crops * n_clips
                # input = input.view(-1, n_views, 3, args.clip_length, input.size(3), input.size(4))
                output, _ = model(  input)  # (batch, n_views, C, T, H, W) ->  (batch, n_class), todo outputs are unnormalized scores
            else:
                input = input.reshape( (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
                output = model(input)  # (batch * n_views, 3, T, 224,224 ) ->  (batch * n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = rearrange(output, '(d0 d1) d2 -> d0 d1 d2', d0=actual_bz)  # (batch * n_views,  n_class) ->  (batch, n_views,  n_class)  todo  reshape clip prediction into video prediction
                output = torch.mean(output, dim=1)  # (batch, n_views,  n_class) ->  (batch,  n_class)


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            _, preds = torch.max(output, 1)
            # pred_concat = np.concatenate([pred_concat, preds.detach().cpu().numpy()])
            # gt_concat = np.concatenate([gt_concat, target.detach().cpu().numpy()])

            # losses.update(loss.item(), actual_bz)
            top1.update(prec1.item(), actual_bz)
            top5.update(prec5.item(), actual_bz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose:
                if i % args.print_freq == 0:
                    logger.debug(('  \tTest Epoch {epoch}: [{0}/{1}]\t'
                                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(eval_loader), epoch=epoch,
                                                                                  batch_time=batch_time, top1=top1,
                                                                                  top5=top5)))

    eval_dua = False
    if args.evaluate_baselines:
        if args.baseline == 'dua':
            eval_dua == True
    if not eval_dua:
            logger.debug(('  \tTesting Results Epoch {epoch}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch,
                                                                                                          top1=top1,
                                                                                                          top5=top5)))
    else:
        logger.debug(
            ('\tTesting Results for DUA after adaptation on video {epoch}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch,
                                                                                                     top1=top1,
                                                                                                     top5=top5)))
        return top1.avg

    if writer is not None:
        # writer.add_scalars('loss', {'val_loss': losses.avg}, global_step=epoch)
        writer.add_scalars('acc', {'test_acc': top1.avg}, global_step=global_iter)
    logger.debug(f'  \tTest Epoch {epoch} acc {top1.avg} ')
    return top1.avg



def validate_old(val_loader, model, criterion, iter, epoch=None, args=None, logger=None, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.reshape(
                (-1,) + input.shape[2:])  # (batch, n_views, 3, T, 224,224 ) -> (batch * n_views, 3, T, 224,224 )
            target = target.reshape((target.shape[0], 1)).repeat(1, args.num_clips)
            target = target.reshape((-1,) + target.shape[2:])

            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(
                1, 5))  # todo this is clip accuracy,  we should take the average of all clip predictions from a video

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.debug(('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)))

    logger.debug(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(top1=top1, top5=top5, loss=losses)))
    writer.add_scalars('loss', {'val_loss': losses.avg}, global_step=epoch)
    writer.add_scalars('acc', {'val_acc': top1.avg}, global_step=epoch)

    return top1.avg