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
    args.dataset = 'somethingv2'
    args.vid_format = '.mp4' # .mp4, .avi

    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ss2/model_swin/swin_base_patch244_window1677_sthv2.pth'

    args.video_data_dir = '/scratch/project_465001897/datasets/ss2/videos/samples_mp4'  
    args.val_vid_list = '/scratch/project_465001897/datasets/ss2/videos/split/train_rgb.txt'
    # todo ========================= To Specify ==========================

    args.batch_size = 12  # 12
    args.num_clips = 1  # number of temporal clips
    args.test_crops = 1  # number of spatial crops
    args.frame_uniform = True
    args.frame_interval = 2
    args.scale_size = 224
    args.patch_size = (2, 4, 4)
    args.clip_length = 16
    args.window_size = (16, 7, 7)

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    # Request DWT subband statistics computation (DWT only)
    args.compute_stat = 'dwt_subbands'
    args.subband_transform = 'dwt'
    args.dwt_wavelet = 'haar'  # always use Haar

    # DWT alignment parameters for stats extraction
    # K levels for deepest subbands; must match what you'll use at TTA time
    args.dwt_align_levels = 1
    # Toggle 3D vs 2D DWT by setting this flag
    args.dwt_align_3d = True  # uncomment to use 3D (T,H,W) DWT

    args.n_augmented_views = None
    args.n_epoch_adapat = 1

    args.corruptions = 'clean'
    # Adapt result_dir based on 2D vs 3D DWT (no wavelet in path)
    base_stats_dir = '/scratch/project_465001897/datasets/ss2'
    subdir = 'source_statistics_swin_dwt3d' if getattr(args, 'dwt_align_3d', False) else 'source_statistics_swin_dwt'
    args.result_dir = os.path.join(base_stats_dir, subdir)
    eval(args=args, )


