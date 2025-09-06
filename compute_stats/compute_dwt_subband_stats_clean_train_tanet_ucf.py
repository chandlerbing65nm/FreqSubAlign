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

    # Request subband statistics computation (DWT/FFT/DCT)
    args.compute_stat = 'dwt_subbands'
    # Use 3D DWT by default for TANet/UCF stats; choose wavelet 'haar' or 'db2'
    args.subband_transform = 'dwt'
    args.dwt_wavelet = 'db2' # haar , db2

    # Subband alignment parameters for stats extraction
    # Enable 3D subbands (T,H,W) for DWT/FFT/DCT
    args.dwt_align_3d = True
    # Validate wavelet for 3D DWT (supports 'haar' and 'db2')
    if args.subband_transform == 'dwt' and getattr(args, 'dwt_align_3d', False):
        wl = str(getattr(args, 'dwt_wavelet', 'haar')).lower()
        if wl not in ['haar', 'db2']:
            print(f"[INFO] 3D DWT supports 'haar' or 'db2'. Got '{wl}', defaulting to 'haar'.")
            args.dwt_wavelet = 'haar'
    # For FFT/DCT, computation is level-1; DWT can use multiple levels
    args.dwt_align_levels = 1

    # Output directory for stats NPZ (organize by transform)
    base_stats_dir = '/scratch/project_465001897/datasets/ucf'
    # Organize outputs into transform-specific directories, with 3D suffix when requested
    subdir = {
        'dwt': (
            f"source_statistics_tanet_dwt3d_{args.dwt_wavelet.lower()}" if getattr(args, 'dwt_align_3d', False)
            else f"source_statistics_tanet_dwt_{args.dwt_wavelet.lower()}"
        ),
        'fft': 'source_statistics_tanet_fft3d' if getattr(args, 'dwt_align_3d', False) else 'source_statistics_tanet_fft',
        'dct': 'source_statistics_tanet_dct3d' if getattr(args, 'dwt_align_3d', False) else 'source_statistics_tanet_dct',
    }.get(args.subband_transform, 'source_statistics_tanet_dwt')
    args.result_dir = os.path.join(base_stats_dir, subdir)

    eval(args=args)
