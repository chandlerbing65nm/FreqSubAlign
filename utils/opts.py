import argparse

# ========================= Constants ==========================
# Normalization constants for TANet
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

# Normalization config for Video Swin Transformer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False
)

# ========================= Parser Setup ==========================
parser = argparse.ArgumentParser(description="ViTTA")

# ========================= Dataset Configuration ==========================
parser.add_argument('--dataset', type=str, default='ucf101',
                    choices=['ucf101', 'somethingv2', 'kinetics'])
parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--root_path', default='None', type=str)
parser.add_argument('--video_data_dir', type=str,
                    default='',
                    help='directory of the corrupted videos')
parser.add_argument('--vid_format', default='', type=str,
                    help='video format if not specified in filenames')
parser.add_argument('--datatype', default='vid', type=str, choices=['vid', 'frame'])
# Choose corruption list size
parser.add_argument('--corruption_list', type=str, default='full', choices=['mini', 'full', 'continual', 'random'],
                    help='Which corruption list to evaluate: mini (quick sanity), full (complete set), continual (mixed sequence), or random (randomized sequence)')

# Optional: print exact validation corruption order as read from list file
parser.add_argument('--print_val_corrupt_order', action='store_true',
                    help='If set, prints the exact order of validation corruption types or entries before evaluation')

# ========================= Statistics Files ==========================
parser.add_argument('--spatiotemp_mean_clean_file', type=str,
                    default='',
                    help='spatiotemporal statistics - mean')
parser.add_argument('--spatiotemp_var_clean_file', type=str,
                    default='',
                    help='spatiotemporal statistics - variance')
parser.add_argument('--val_vid_list', type=str,
                    default='',
                    help='list of corrupted videos to adapt to')
parser.add_argument('--result_dir', type=str,
                    default='',
                    help='result directory')
parser.add_argument('--result_suffix', type=str, default='',
                    help='custom suffix to append to result files for unique identification')

# ========================= Model Configuration ==========================
parser.add_argument('--arch', type=str, default='tanet',
                    choices=['tanet', 'videoswintransformer'],
                    help='network architecture')
parser.add_argument('--model_path', type=str,
                    default='')
parser.add_argument('--img_feature_dim', type=int, default=256,
                    help='dimension of image feature on ResNet50')
parser.add_argument('--partial_bn', action='store_true')

# ========================= Video Swin Transformer Specific Configs ==========================
parser.add_argument('--num_clips', type=int, default=1,
                    help='number of temporal clips')
parser.add_argument('--frame_uniform', type=bool, default=True,
                    help='whether uniform sampling or dense sampling')
parser.add_argument('--frame_interval', type=int, default=2)
parser.add_argument('--flip_ratio', type=int, default=0)
parser.add_argument('--img_norm_cfg', default=img_norm_cfg)
parser.add_argument('--patch_size', default=(2,4,4))
parser.add_argument('--window_size', default=(8, 7, 7))
parser.add_argument('--drop_path_rate', default=0.2)

# ========================= Runtime Configuration ==========================
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--debug', action='store_true',
                    help='if debug, loading only the first 50 videos in the list')
parser.add_argument('--verbose', type=bool, default=True,
                    help='more details in the logging file')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    help='print frequency (default: 5)')

# ========================= Test-Time Adaptation Configuration ==========================
parser.add_argument('--tta', type=bool, default=True,
                    help='perform test-time adaptation')
parser.add_argument('--phase_only_preprocessing', type=bool, default=False,
                    help='apply Fourier transform to retain only phase information')
parser.add_argument('--dwt_preprocessing', type=bool, default=False,
                    help='apply Discrete Wavelet Transform preprocessing')
parser.add_argument('--dwt_component', type=str, default='LL',
                    help='DWT component(s) to use for reconstruction (LL, LH, HL, HH or combinations like LL+LH)')
parser.add_argument('--dwt_levels', type=int, default=1,
                    help='Number of DWT decomposition levels (K>=1). Components are selected at the deepest level.')
# ========================= DWT Subband Alignment (Hook) ==========================
parser.add_argument('--dwt_align_enable', action='store_true',
                    help='Enable DWT subband statistics alignment hook')
parser.add_argument('--subband_transform', type=str, default='dwt', choices=['dwt', 'fft', 'dct'],
                    help='Subband transform to use for stats and alignment: dwt (default), fft, or dct. FFT/DCT are 2D level-1 only and heuristically partitioned into LL/LH/HL/HH.')
parser.add_argument('--dwt_align_adaptive_lambda', action='store_true', 
                    help='Scale lambdas by subband energy proportion')
parser.add_argument('--dwt_align_levels', type=int, default=1,
                    help='DWT decomposition levels for alignment (deepest level used)')
parser.add_argument('--dwt_align_3d', type=bool, default=False,
                    help='If True, apply 3D DWT over (T,H,W); otherwise apply 2D DWT over (H,W) per frame')
parser.add_argument('--dwt_align_lambda_ll', type=float, default=1.0,
                    help='Lambda weight for LL subband alignment')
parser.add_argument('--dwt_align_lambda_lh', type=float, default=1.0,
                    help='Lambda weight for LH subband alignment')
parser.add_argument('--dwt_align_lambda_hl', type=float, default=1.0,
                    help='Lambda weight for HL subband alignment')
parser.add_argument('--dwt_align_lambda_hh', type=float, default=1.0,
                    help='Lambda weight for HH subband alignment')
parser.add_argument('--dwt_stats_npz_file', type=str, default='',
                    help='Path to NPZ file containing clean DWT subband stats per chosen layer: keys like LL_mean, LL_var, LH_mean, LH_var, HL_mean, HL_var, HH_mean, HH_var')
# Global weights to combine base (ViTTA) vs DWT alignment losses
parser.add_argument('--lambda_base_align', type=float, default=1.0,
                    help='Weight for base (ViTTA) alignment loss; set 0 to disable base alignment')
parser.add_argument('--lambda_dwt_align', type=float, default=1.0,
                    help='Weight for DWT subband alignment loss; set 0 to disable DWT alignment')
parser.add_argument('--use_src_stat_in_reg', type=bool, default=True,
                    help='whether to use source statistics in the regularization loss')
parser.add_argument('--fix_BNS', type=bool, default=True,
                    help='whether fix the BNS of target model during forward pass')
parser.add_argument('--running_manner', type=bool, default=True,
                    help='whether to manually compute the target statistics in running manner')
parser.add_argument('--momentum_bns', type=float, default=0.1)
parser.add_argument('--update_only_bn_affine', action='store_true')
parser.add_argument('--compute_stat', action='store_true')
parser.add_argument('--momentum_mvg', type=float, default=0.1)
parser.add_argument('--stat_reg', type=str, default='mean_var',
                    help='statistics regularization')
parser.add_argument('--if_tta_standard', type=str, default='tta_online')
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])

# ========================= Augmentation Configuration ==========================
parser.add_argument('--if_sample_tta_aug_views', type=bool, default=True)
parser.add_argument('--if_spatial_rand_cropping', type=bool, default=True)
parser.add_argument('--if_pred_consistency', type=bool, default=True)
parser.add_argument('--lambda_pred_consis', type=float, default=0.1)
parser.add_argument('--lambda_feature_reg', type=int, default=1)
parser.add_argument('--include_ce_in_consistency', type=bool, default=False,
                    help='Whether to include cross-entropy loss when using prediction consistency')
parser.add_argument('--n_augmented_views', type=int, default=2)
parser.add_argument('--tta_view_sample_style_list', default=['uniform_equidist'])
parser.add_argument('--stat_type', default=['spatiotemp'])
parser.add_argument('--before_norm', action='store_true')
parser.add_argument('--reduce_dim', type=bool, default=True)
parser.add_argument('--reg_type', type=str, default='l1_loss')
parser.add_argument('--chosen_blocks', default=['layer3', 'layer4'])
parser.add_argument('--moving_avg', type=bool, default=True)
parser.add_argument('--n_gradient_steps', type=int, default=1,
                    help='number of gradient steps per sample')

# ========================= Input Configuration ==========================
parser.add_argument('--full_res', action='store_true')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--scale_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--clip_length', type=int, default=16)
parser.add_argument('--sample_style', type=str, default='uniform-1',
                    help="either 'dense-xx' (dense sampling) or 'uniform-xx' (uniform sampling, TSN style)")
parser.add_argument('--test_crops', type=int, default=1,
                    help="number of spatial crops")
parser.add_argument('--use_pretrained', action='store_true',
                    help='whether to use pretrained model for training')
parser.add_argument('--input_mean', default=input_mean)
parser.add_argument('--input_std', default=input_std)

# ========================= Training Configuration ==========================
parser.add_argument('--lr', default=0.00005)
parser.add_argument('--n_epoch_adapat', type=int, default=1,
                    help='number of adaptation epochs for TTA (default: 1 for backward compatibility)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

def get_opts():
    args = parser.parse_args()
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'
    return args