# import os.path as osp
# from utils.opts import parser
# import argparse
from utils.opts import get_opts
from utils.utils_ import get_writer_to_all_result
from corpus.main_eval import eval


corruptions = [
    'gauss_shuffled', 'pepper_shuffled', 'salt_shuffled', 'shot_shuffled',
    'zoom_shuffled', 'impulse_shuffled', 'defocus_shuffled', 'motion_shuffled',
    'jpeg_shuffled', 'contrast_shuffled', 'rain_shuffled', 'h265_abr_shuffled',  
    ]

if __name__ == '__main__':
    global args
    args = get_opts()
    args.gpus = [0]
    args.arch = 'tanet'
    args.dataset = 'ucf101'
    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet_ucf/tanet_ucf.pth.tar'
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/val_corruptions' #  main directory of the video data,  [args.video_data_dir] + [path in file list] should be complete absolute path for a video file
    args.spatiotemp_mean_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet_ucf/list_spatiotemp_mean_20220908_235138.npy'
    args.spatiotemp_var_clean_file = '/scratch/project_465001897/datasets/ucf/source_statistics_tanet_ucf/list_spatiotemp_var_20220908_235138.npy'
    # todo ========================= To Specify ==========================

    args.batch_size = 8

    for corr_id, args.corruptions in enumerate(corruptions):
        print(f'####Starting Evaluation for ::: {args.corruptions} corruption####')
        args.val_vid_list = f'/scratch/project_465001897/datasets/ucf/list_video_perturbations_ucf/{args.corruptions}.txt'
        args.result_dir = f'/scratch/project_465001897/datasets/ucf/results/{args.arch}/{args.dataset}/{args.corruptions}'

        epoch_result_list, _ = eval(args=args, )

        if corr_id == 0:
            f_write = get_writer_to_all_result(args)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in epoch_result_list]) + '\n')

        f_write.flush()
        if corr_id == len(corruptions) - 1:
            f_write.close()
