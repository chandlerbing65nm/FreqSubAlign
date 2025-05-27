import os
# Configure AMD GPU
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['HIP_LAUNCH_BLOCKING'] = '1'  # For better error reporting

from utils.opts import get_opts
from utils.utils_ import get_writer_to_all_result
from corpus.main_eval import eval
import torch

# Set device configuration
torch.backends.cudnn.enabled = False  # Disable cuDNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corruptions = [
    'gauss_shuffled', 'pepper_shuffled', 'salt_shuffled', 'shot_shuffled',
    'zoom_shuffled', 'impulse_shuffled', 'defocus_shuffled', 'motion_shuffled',
    'jpeg_shuffled', 'contrast_shuffled', 'rain_shuffled', 'h265_abr_shuffled',  
]

def get_model_config(arch):
    config = {
        'model_path': '',
        'spatiotemp_mean_clean_file': '',
        'spatiotemp_var_clean_file': '',
        'additional_args': {}
    }
    
    if arch == 'videoswintransformer':
        config.update({
            'model_path': '/scratch/project_465001897/datasets/ucf/model_swin_ucf/swin_ucf_base_patch244_window877_pretrain_kinetics400_30epoch_lr3e-5.pth',
            'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_swin_ucf/list_spatiotemp_mean_20221004_192722.npy',
            'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_swin_ucf/list_spatiotemp_var_20221004_192722.npy',
            'additional_args': {
                'clip_length': 16,
                'num_clips': 1,
                'test_crops': 1,
                'frame_uniform': True,
                'frame_interval': 2,
                'scale_size': 224,
                'patch_size': (2,4,4),
                'window_size': (8, 7, 7),
                'lr': 0.00001,
                'lambda_pred_consis': 0.05,
                'momentum_mvg': 0.05,
                'chosen_blocks': ['module.backbone.layers.2', 'module.backbone.layers.3', 'module.backbone.norm']
            }
        })
    elif arch == 'tanet':
        config.update({
            'model_path': '/scratch/project_465001897/datasets/ucf/model_tanet_ucf/tanet_ucf.pth.tar',
            'spatiotemp_mean_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_tanet_ucf/list_spatiotemp_mean_20220908_235138.npy',
            'spatiotemp_var_clean_file': '/scratch/project_465001897/datasets/ucf/source_statistics_tanet_ucf/list_spatiotemp_var_20220908_235138.npy',
            'additional_args': {}
        })
    
    return config

if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.dataset = 'ucf101'
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/val_corruptions'
    args.batch_size = 8

    # Choose model architecture (either 'videoswintransformer' or 'tanet')
    args.arch = 'videoswintransformer'  # Change this to switch between models
    
    # Get model-specific configuration
    model_config = get_model_config(args.arch)
    args.model_path = model_config['model_path']
    args.spatiotemp_mean_clean_file = model_config['spatiotemp_mean_clean_file']
    args.spatiotemp_var_clean_file = model_config['spatiotemp_var_clean_file']
    
    # Set additional arguments for Swin Transformer if needed
    for key, value in model_config['additional_args'].items():
        setattr(args, key, value)

    for corr_id, args.corruptions in enumerate(corruptions):
        print(f'####Starting Evaluation for ::: {args.corruptions} corruption####')
        args.val_vid_list = f'/scratch/project_465001897/datasets/ucf/list_video_perturbations_ucf/{args.corruptions}.txt'
        args.result_dir = f'/scratch/project_465001897/datasets/ucf/results/{args.arch}_{args.dataset}/tta_{args.corruptions}'

        epoch_result_list, _ = eval(args=args)

        if corr_id == 0:
            f_write = get_writer_to_all_result(args)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in epoch_result_list]) + '\n')

        f_write.flush()
        if corr_id == len(corruptions) - 1:
            f_write.close() 