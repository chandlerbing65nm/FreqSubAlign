import os
import sys
sys.path.append(os.path.abspath('..'))    # last level
# import os.path as osp
# from utils.opts import parser
from utils.opts import get_opts
from corpus.main_eval import eval

corruptions = ['clean' ]

if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'videoswintransformer'
    args.dataset = 'ucf101'
    args.vid_format = '.avi' # .webm, .avi

    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_swin/swin_ucf_base_patch244_window877_pretrain_kinetics400_30epoch_lr3e-5.pth'
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/videos/samples'  
    args.val_vid_list = '/scratch/project_465001897/datasets/ucf/videos/split/train_rgb_split_1.txt'
    # todo ========================= To Specify ==========================

    args.batch_size = 12  # 12
    args.num_clips = 1  # number of temporal clips
    args.test_crops = 1  # number of spatial crops
    args.frame_uniform = True
    args.frame_interval = 2
    args.scale_size = 224
    args.patch_size = (2, 4, 4)
    args.clip_length = 16
    args.window_size = (8, 7, 7)

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    # Request DWT subband statistics computation
    args.compute_stat = 'dwt_subbands'

    # DWT alignment parameters for stats extraction
    # K levels for deepest subbands; must match what you'll use at TTA time
    args.dwt_align_levels = 1

    args.n_augmented_views = None
    args.n_epoch_adapat = 1

    args.corruptions = 'clean'
    args.result_dir = f'/scratch/project_465001897/datasets/ucf/source_statistics_swin_dwt'
    eval(args=args, )


