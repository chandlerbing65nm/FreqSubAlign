import torch.nn as nn
from utils.utils_ import *
# from corpus.main_train import validate_brief
from corpus.training import train, validate, validate_brief
from baselines.dua_utils import rotate_batch
from utils.pap_tcsm import run_pap_warmup


def DUA(model):
    model = configure_model(model)
    return model


def configure_model(model):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
    return model


def dua_adaptation(args, model, te_loader, adapt_loader, logger, batchsize, augmentations, no_vids):
    """
    :param model: After configuring the DUA model
    :param te_loader: The test set for Test-Time-Training
    :param logger: Logger for logging results
    :param batchsize: Batchsize to use for adaptation
    :param augmentations: augmentations to form a batch from a single video
    :param no_vids: total number of videos for adaptation

    """
    if args.arch == 'tanet':
        from models.tanet_models.transforms import ToTorchFormatTensor_TANet_dua, GroupNormalize_TANet_dua
        adapt_transforms = torchvision.transforms.Compose([
            augmentations,  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip
            ToTorchFormatTensor_TANet_dua(div=True),
            GroupNormalize_TANet_dua(args.input_mean, args.input_std)
        ])
    else:
        adapt_transforms = torchvision.transforms.Compose([
            augmentations,  # GroupMultiScaleCrop  amd   GroupRandomHorizontalFlip
            fromListToTorchFormatTensor(clip_len=args.clip_length, num_clips=args.num_clips),
            GroupNormalize(args.input_mean, args.input_std)
            # Normalize later in the DUA adaptation loop after making a batch
        ])
    logger.debug('---- Starting adaptation for DUA ----')
    did_pap = False
    all_acc = []
    for i, (inputs, target) in enumerate(adapt_loader):
        # Optional PAP/TCSM warm-up once before adaptation loop
        if (not did_pap) and getattr(args, 'probe_ffp_enable', False):
            if args.arch == 'tanet':
                x_probe = inputs.cuda()
                actual_bz_probe = x_probe.shape[0]
                x_probe = x_probe.view(-1, 3, x_probe.size(2), x_probe.size(3))
                frames_total_probe = x_probe.shape[0]
                frames_per_sample_probe = frames_total_probe // actual_bz_probe
                num_clips_probe = max(1, frames_per_sample_probe // args.clip_length)
                x_probe = x_probe.view(actual_bz_probe * num_clips_probe,
                                       args.clip_length, 3, x_probe.size(2), x_probe.size(3))
            else:
                x_probe = inputs.cuda()
                x_probe = x_probe.reshape((-1,) + x_probe.shape[2:])
            _ = run_pap_warmup(model, x_probe, args, logger)
            did_pap = True
        model.train()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()

        with torch.no_grad():
            if args.arch == 'tanet':
                inputs = inputs.cuda()
                actual_bz = inputs.shape[0]

                # inputs comes as [B, 3*T*(crops? clips?), H, W] after Stack_TANet.
                # First, split channel-time into frames: [B*T*(crops? clips?), 3, H, W]
                inputs = inputs.view(-1, 3, inputs.size(2), inputs.size(3))
                # Infer frames-per-sample and number of clips to avoid mismatches when test_crops are absent in adapt_loader
                frames_total = inputs.shape[0]
                frames_per_sample = frames_total // actual_bz
                num_clips_calc = max(1, frames_per_sample // args.clip_length)
                inputs = inputs.view(actual_bz * num_clips_calc,
                                     args.clip_length, 3, inputs.size(2), inputs.size(3))  # [*, 16, 3, H, W]
                inputs = [(adapt_transforms([inputs, target])[0]) for _ in
                          range(batchsize)]  # pass image, label together
                inputs = torch.stack(inputs)  # only stack images
                inputs = inputs.cuda()
                rot_img = rotate_batch(inputs)
                _ = model(rot_img.float())
            else:
                inputs = [(adapt_transforms([inputs, target])[0]) for _ in
                          range(batchsize)]  # pass image, label together
                inputs = torch.stack(inputs)  # only stack images
                inputs = inputs.cuda()
                inputs = inputs.reshape(
                    (-1,) + inputs.shape[2:])  # [b, channel, frames, h, w]
                rot_img = rotate_batch(inputs)
                _ = model(rot_img)

            logger.debug(f'---- Starting evaluation for DUA after video {i} ----')

        if i % 1 == 0 or i == len(adapt_loader) - 1:
            top1 = validate_brief(eval_loader=te_loader, model=model, global_iter=i, args=args,
                                  logger=logger, writer=None, epoch=i)
            all_acc.append(top1)

        if len(all_acc) >= 3:
            if all(top1 < i for i in all_acc[-3:]):
                logger.debug('---- Model Performance Degrading Consistently ::: Quitting Now ----')
                return max(all_acc)

        if i == no_vids:
            logger.debug(f' --- Best Accuracy for {args.corruptions} --- {max(all_acc)}')
            logger.debug(f' --- Stopping DUA adaptation ---')
            return max(all_acc)

    return max(all_acc)


