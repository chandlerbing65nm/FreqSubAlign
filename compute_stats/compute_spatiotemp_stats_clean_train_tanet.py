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
    args.dataset = 'uffia' # somethingv2, ucf101
    args.vid_format = '.mp4' # .webm, .avi


    # todo ========================= To Specify ==========================
    args.model_path = '/scratch/project_465001897/datasets/ucf/model_tanet/tanet_ucf.pth.tar'
    args.img_feature_dim = 256  # UCF-101 TANet uses 256
    args.video_data_dir = '/scratch/project_465001897/datasets/uffia/video' # Use fish dataset videos
    args.val_vid_list = '/scratch/project_465001897/datasets/uffia/split/train_rgb_split_1.txt' # Use fish training data for statistics
    # todo ========================= To Specify ==========================

    args.clip_length = 8
    args.batch_size = 12  # 12
    args.sample_style = 'uniform-1'  # number of temporal clips
    args.test_crops = 1  # number of spatial crops

    args.tta = True
    args.evaluate_baselines = not args.tta
    args.baseline = 'source'

    args.compute_stat = 'mean_var'
    args.stat_type = 'spatiotemp' # temp, spatiotemp

    args.corruptions = 'clean'
    args.result_dir = f'/scratch/project_465001897/datasets/uffia/source_statistics_tanet'
    eval(args=args, )


