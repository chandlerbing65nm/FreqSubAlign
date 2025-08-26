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
    args.test_crops = 1
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
    args.result_dir = f'/scratch/project_465001897/datasets/ss2/source_statistics_tanet'
    eval(args=args, )


