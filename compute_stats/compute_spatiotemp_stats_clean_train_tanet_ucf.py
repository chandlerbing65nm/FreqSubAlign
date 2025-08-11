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
    args.dataset = 'ucf101' # somethingv2, ucf101, uffia
    args.vid_format = '.avi' # .webm, .avi


    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'

    args.video_data_dir = '/scratch/project_465001897/datasets/ucf/videos/samples' # Use fish dataset videos
    args.val_vid_list = '/scratch/project_465001897/datasets/ucf/videos/split/train_rgb_split_1.txt' # Use fish training data for statistics
    # todo ========================= To Specify ==========================

    args.clip_length = 16
    args.test_crops = 3
    args.num_clips = 1
    args.batch_size = 24
    args.scale_size = 256
    # args.crop_size = 256
    args.input_size = 224
    args.frame_uniform = True
    args.frame_interval = 2
    args.sample_style = 'uniform-1'  # TANet: single clip

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    args.compute_stat = 'mean_var'
    args.stat_type = 'spatiotemp' # temp, spatiotemp

    args.corruptions = 'clean'
    args.result_dir = f'/scratch/project_465001897/datasets/ucf/source_statistics_tanet'
    eval(args=args, )


