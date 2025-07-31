import os
import sys
sys.path.append(os.path.abspath('..'))    # last level
# import os.path as osp
# from utils.opts import parser
from utils.opts import get_opts
from corpus.main_eval import eval

corruptions = ['clean' ]

def set_architecture_params(args):
    """Automatically set parameters based on architecture and dataset."""
    if args.arch == 'videoswintransformer':
        if args.dataset == 'somethingv2':
            args.clip_length = 16
            args.window_size = (16, 7, 7)
        elif args.dataset == 'ucf101':
            args.clip_length = 16
            args.window_size = (8, 7, 7)
        elif args.dataset == 'uffia':
            args.clip_length = 16
            args.window_size = (8, 7, 7)
    elif args.arch == 'tanet':
        if args.dataset == 'somethingv2':
            args.clip_length = 8
            args.window_size = (8, 7, 7)
        elif args.dataset == 'ucf101':
            args.clip_length = 16
            args.window_size = (8, 7, 7)
        elif args.dataset == 'uffia':
            args.clip_length = 16
            args.window_size = (8, 7, 7)
    return args

if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'videoswintransformer'
    args.dataset = 'uffia'
    args.vid_format = '.mp4' # .webm, .avi

    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_swin/swin_ucf_base_patch244_window877_pretrain_kinetics400_30epoch_lr3e-5.pth'
    args.video_data_dir = '/scratch/project_465001897/datasets/uffia/video'  #  main directory of the video data,  [args.video_data_dir] + [path in file list] should be complete absolute path for a video file
    args.val_vid_list = '/scratch/project_465001897/datasets/uffia/split/train_rgb_split_1.txt' # list of training data for computing statistics, with lines in format :   file_path n_frames class_id
    # todo ========================= To Specify ==========================

    args.batch_size = 12  # 12
    args.num_clips = 1  # number of temporal clips
    args.test_crops = 1  # number of spatial crops
    args.frame_uniform = True
    args.frame_interval = 2
    args.scale_size = 224
    args.patch_size = (2, 4, 4)

    # Automatically set architecture-specific parameters
    args = set_architecture_params(args)

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    args.n_augmented_views = None
    args.n_epoch_adapat = 1

    args.compute_stat = 'mean_var'
    args.stat_type = 'spatiotemp'

    args.corruptions = 'clean'
    args.result_dir = f'/scratch/project_465001897/datasets/uffia/source_statistics_swin'
    eval(args=args, )


