import os
import sys
print(os.getcwd())
sys.path.append(os.path.abspath('..'))
from utils.opts import get_opts
from corpus.main_eval import eval


if __name__ == '__main__':
    args = get_opts()
    # Hardware / arch / dataset
    args.gpus = [0]
    args.arch = 'tanet'
    args.dataset = 'ucf101'  # somethingv2, ucf101, kinetics
    args.modality = 'RGB'
    args.vid_format = '.avi'

    # ========================= To Specify ==========================
    # Pretrained TANet checkpoint path
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'

    # Clean TRAIN split (videos and list) to compute stats on
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/videos/samples'
    args.val_vid_list = '/scratch/project_465001897/datasets/ucf/videos/split/train_rgb_split_1.txt'
    # ========================= To Specify ==========================

    # Inference/data params
    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.batch_size = 12
    args.scale_size = 256
    args.input_size = 224
    args.frame_uniform = True
    args.frame_interval = 2
    args.sample_style = 'uniform-1'  # TANet: single temporal clip per video

    # TTA/eval driver settings
    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    # Request DWT subband statistics computation
    args.compute_stat = 'dwt_subbands'

    # DWT alignment parameters for stats extraction
    # args.dwt_align_3d = True
    args.dwt_align_levels = 1

    # Output directory for stats NPZ
    args.result_dir = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet_dwt'

    eval(args=args)
