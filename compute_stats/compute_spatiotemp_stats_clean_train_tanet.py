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
    args.dataset = 'ucf101'
    args.vid_format = '.avi'


    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet_ucf/tanet_ucf.pth.tar'
    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/videos/samples' #  main directory of the video data,  [args.video_data_dir] + [path in file list] should be complete absolute path for a video file
    args.val_vid_list = '/scratch/project_465001897/datasets/ucf/videos/split/train_rgb_split_1.txt' # list of training data for computing statistics, with lines in format :   file_path n_frames class_id
    # todo ========================= To Specify ==========================

    args.clip_length = 16
    args.batch_size = 12  # 12
    args.sample_style = 'uniform-1'  # number of temporal clips
    args.test_crops = 1  # number of spatial crops

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    args.compute_stat = 'mean_var'
    args.stat_type = 'spatiotemp' # temp, spatiotemp

    args.corruptions = 'clean'
    args.result_dir = f'/scratch/project_465001897/datasets/ucf/source_statistics_tanet_ucf'
    eval(args=args, )


