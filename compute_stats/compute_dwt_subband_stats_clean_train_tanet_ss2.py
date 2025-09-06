import os
import sys
print(os.getcwd())
sys.path.append(os.path.abspath('..'))
from utils.opts import get_opts
from corpus.main_eval import eval


if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'tanet'
    args.dataset = 'somethingv2' # somethingv2, ucf101, uffia
    args.vid_format = '.mp4' # .webm, .avi


    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ss2/model_tanet/TR50_S2_256_8x3x2.pth.tar'

    args.video_data_dir = '/scratch/project_465001897/datasets/ss2/videos/samples_mp4'
    args.val_vid_list = '/scratch/project_465001897/datasets/ss2/videos/split/train_rgb.txt' # Use fish training data for statistics
    # todo ========================= To Specify ==========================

    args.clip_length = 8
    args.test_crops = 3
    args.num_clips = 1
    args.batch_size = 12
    args.scale_size = 256
    # args.crop_size = 256
    args.input_size = 224
    args.frame_uniform = True
    args.frame_interval = 2
    args.sample_style = 'uniform-1'  # TANet: single clip
    args.stat_type = 'spatiotemp' # temp, spatiotemp

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

    args.corruptions = 'clean'
    # Adapt result_dir based on 2D vs 3D DWT (no wavelet in path)
    base_stats_dir = '/scratch/project_465001897/datasets/ss2'
    subdir = 'source_statistics_tanet_dwt3d' if getattr(args, 'dwt_align_3d', False) else 'source_statistics_tanet_dwt'
    args.result_dir = os.path.join(base_stats_dir, subdir)
    eval(args=args, )


