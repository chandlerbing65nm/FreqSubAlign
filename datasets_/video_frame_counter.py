""" video_frame_counter.py
Combines functionality of video2image.py and gen_label.py but only counts frames
"""
from __future__ import division, print_function

import argparse
import glob
import multiprocessing
import os
import os.path as osp
import subprocess
from functools import partial
import json
import shutil

n_thread = 50

def parse_args():
    """Parse arguments for frame counting and label generation"""
    parser = argparse.ArgumentParser(description='Count frames in videos and generate labels')
    parser.add_argument('video_path', type=str,
                        help='root directory for the input videos')
    parser.add_argument('out_path', type=str,
                        help='root directory for the output labels')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3, 4],
                        help='the number of level for folders')
    parser.add_argument('--lib', type=str, default='ffmpeg',
                        choices=['opencv', 'ffmpeg'],
                        help='the decode lib')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['uffia', 'ucf101', 'hmdb51', 'kinetics400', 'kinetics600', 'kinetics700', 'sthv1', 'sthv2'],
                        help='name of the dataset')
    parser.add_argument('--ann_root', type=str, default='annotation',
                        help='root directory for annotations')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'val'])
    parser.add_argument('--split', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    return args

def count_frames(video_path, decode_type='ffmpeg'):
    """Count number of frames in a video file
    
    Args:
        video_path (str): Path to video file
        decode_type (str): Library to use for decoding ('ffmpeg' or 'opencv')
    
    Returns:
        int: Number of frames in the video
    """
    if decode_type == 'ffmpeg':
        cmd = f'ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "{video_path}"'
        try:
            frame_count = int(subprocess.check_output(cmd, shell=True))
            return frame_count
        except Exception as e:
            return 0
    else:  # opencv
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

def process_video(tup, decode_type, video_root=None):
    """Process a single video to count frames
    
    Args:
        tup (tuple): (source_path, dest_path)
        decode_type (str): Library to use for decoding
        video_root (str): Root video directory to calculate relative path
    
    Returns:
        tuple: (relative_path, frame_count)
    """
    src, dest = tup
    if video_root:
        # Generate relative path from video root (for level > 1)
        video_id = os.path.relpath(src, video_root)
        video_id = os.path.splitext(video_id)[0]  # Remove extension
    else:
        # Just use filename (for level 1)
        video_id = os.path.splitext(os.path.basename(src))[0]
    frame_count = count_frames(src, decode_type)
    return (video_id, frame_count)

def parse_label_file(file):
    """Parse annotation file to get category mapping"""
    categories = []
    with open(file) as f:
        if file.endswith('json'):
            data = json.load(f)
            for i, (cat, idx) in enumerate(data.items()):
                assert i == int(idx)
                categories.append(cat)
        elif 'kinetics' in file:
            lines = f.readlines()
            categories = [c.strip().replace(' ', '_').replace(
                '"', '').replace('(', '').replace(
                ')', '').replace("'", '') for c in lines]
        else:
            lines = f.readlines()
            categories = [line.rstrip() for line in lines]

    if 'sthv1' in file:
        categories = sorted(categories)
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    return dict_categories

def generate_labels(video_path, out_path, frame_counts, dataset, ann_path, level=2, phase='train', split=1):
    """Generate label files using frame counts
    
    Args:
        video_path (str): Path to video directory
        out_path (str): Path to output directory
        frame_counts (dict): Dictionary mapping video paths to frame counts
        dataset (str): Dataset name
        ann_path (str): Path to annotations
        level (int): Directory level
        phase (str): 'train' or 'val'
        split (int): Split number for UCF101/HMDB51
    """
    if not osp.exists(out_path):
        os.makedirs(out_path)

    if 'sth' in dataset:
        generate_sth_labels(video_path, ann_path, out_path, frame_counts)
    elif 'kinetics' in dataset:
        generate_kinetics_labels(video_path, ann_path, out_path, frame_counts, level, phase)
    elif dataset == 'uffia':
        generate_uffia_labels(video_path, ann_path, out_path, frame_counts, split)
    elif dataset in ['ucf101', 'hmdb51']:
        generate_ucf_hmdb_labels(video_path, ann_path, out_path, frame_counts, split)

def generate_sth_labels(data_path, ann_path, out_path, frame_counts):
    """Generate labels for Something-Something dataset"""
    if 'sthv1' in ann_path:
        dataset_name = 'something-something-v1'
        label_file = osp.join(ann_path, '%s-labels.csv' % dataset_name)
        files_input = [osp.join(ann_path, '%s-validation.csv' % dataset_name),
                      osp.join(ann_path, '%s-train.csv' % dataset_name)]
        files_output = [osp.join(out_path, 'val_rgb.txt'),
                       osp.join(out_path, 'train_rgb.txt')]
    else:  # sthv2
        dataset_name = 'something-something-v2'
        label_file = osp.join(ann_path, '%s-labels.json' % dataset_name)
        files_input = [osp.join(ann_path, '%s-validation.json' % dataset_name),
                      osp.join(ann_path, '%s-train.json' % dataset_name),
                      osp.join(ann_path, '%s-test.json' % dataset_name)]
        files_output = [osp.join(out_path, 'val_rgb.txt'),
                       osp.join(out_path, 'train_rgb.txt'),
                       osp.join(out_path, 'test_rgb.txt')]

    dict_categories = parse_label_file(label_file)
    
    for filename_input, filename_output in zip(files_input, files_output):
        output = []
        with open(filename_input) as f:
            if filename_input.endswith('json'):
                data = json.load(f)
                for item in data:
                    video_id = item['id']
                    if 'test' not in filename_input:
                        category = item['template'].replace('[', '').replace(']', '')
                        category_idx = dict_categories[category]
                    else:
                        category_idx = 0
                    
                    frame_count = frame_counts.get(video_id, 0)
                    output.append(f'{video_id} {frame_count} {category_idx}')
            else:  # csv
                lines = f.readlines()
                for line in lines:
                    items = line.rstrip().split(';')
                    video_id = items[0]
                    category = items[1]
                    category_idx = dict_categories[category]
                    
                    frame_count = frame_counts.get(video_id, 0)
                    output.append(f'{video_id} {frame_count} {category_idx}')
        
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))

def generate_kinetics_labels(data_path, ann_path, out_path, frame_counts, level=1, phase='train'):
    """Generate labels for Kinetics dataset"""
    if '400' in ann_path:
        num_class = 400
    elif '600' in ann_path:
        num_class = 600
    elif '700' in ann_path:
        num_class = 700

    label_file = osp.join(ann_path, f'kinetics-{num_class}_label_map.txt')
    file_input = osp.join(ann_path, f'kinetics-{num_class}_{phase}.csv')
    file_out = f'kinetics_rgb_{phase}.txt'

    dict_categories = parse_label_file(label_file)
    assert len(dict_categories.keys()) == num_class

    output = []
    missing_videos = []
    
    with open(file_input) as f:
        lines = f.readlines()[1:]
    
    for line in lines:
        items = line.rstrip().split(',')
        video_id = f"{items[1]}_{int(items[2]):06d}_{int(items[3]):06d}"
        category = items[0].replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '')
        category_idx = dict_categories[category]
        
        if level == 1:
            sub_dir = video_id
        else:  # level == 2
            sub_dir = osp.join(category, video_id)
        
        frame_count = frame_counts.get(sub_dir, 0)
        if frame_count > 0:
            output.append(f'{sub_dir} {frame_count} {category_idx}')
        else:
            missing_videos.append(sub_dir)
    
    with open(osp.join(out_path, file_out), 'w') as f:
        f.write('\n'.join(output))
    with open(osp.join(out_path, f'missing_{file_out}'), 'w') as f:
        f.write('\n'.join(missing_videos))

def generate_ucf_hmdb_labels(data_path, ann_path, out_path, frame_counts, split):
    """Generate labels for UCF101 and HMDB51 datasets"""
    label_file = osp.join(ann_path, 'category.txt')
    files_input = [osp.join(ann_path, f'trainlist0{split}.txt'),
                #   osp.join(ann_path, f'testlist0{split}.txt')]
                    osp.join(ann_path, f'vallist0{split}.txt')]
    files_output = [osp.join(out_path, f'train_rgb_split_{split}.txt'),
                   osp.join(out_path, f'val_rgb_split_{split}.txt')]

    dict_categories = parse_label_file(label_file)
    
    for filename_input, filename_output in zip(files_input, files_output):
        output = []
        with open(filename_input) as f:
            lines = f.readlines()
        
        for line in lines:
            label, name = osp.split(line.rstrip())
            video_id = name.split('.')[0]
            category_idx = dict_categories[label]
            
            frame_count = frame_counts.get(osp.join(label, video_id), 0)
            output.append(f'{label}/{video_id} {frame_count} {category_idx}')
        
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))

def generate_uffia_labels(data_path, ann_path, out_path, frame_counts, split):
    """Generate labels for UFFIA dataset"""
    # Create category mapping for UFFIA classes
    categories = ['none', 'strong', 'medium', 'weak']
    dict_categories = {cat: i for i, cat in enumerate(categories)}
    
    # Write category file
    category_file = osp.join(out_path, 'category.txt')
    with open(category_file, 'w') as f:
        for cat in categories:
            f.write(f'{cat}\n')
    
    files_input = [osp.join(ann_path, f'trainlist0{split}.txt'),
                   osp.join(ann_path, f'vallist0{split}.txt')]
    files_output = [osp.join(out_path, f'train_rgb_split_{split}.txt'),
                   osp.join(out_path, f'val_rgb_split_{split}.txt')]

    for filename_input, filename_output in zip(files_input, files_output):
        output = []
        with open(filename_input) as f:
            lines = f.readlines()
        
        for line in lines:
            # UFFIA format: 2022_6_13/AM_40/none/13_video_1.mp4
            video_path = line.rstrip()
            video_id = video_path.replace('.mp4', '')
            
            # Extract class from path (3rd level: date/time/class/file)
            path_parts = video_id.split('/')
            if len(path_parts) >= 3:
                class_name = path_parts[2]  # 'none', 'weak', 'medium', 'strong'
                category_idx = dict_categories.get(class_name, 0)
            else:
                category_idx = 0
            
            frame_count = frame_counts.get(video_id, 0)
            output.append(f'{video_id} {frame_count} {category_idx}')
        
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))

def calculate_split_totals(frame_counts, dataset, ann_path, phase='train', split=1):
    """Calculate total frames for each split
    
    Args:
        frame_counts (dict): Dictionary mapping video paths to frame counts
        dataset (str): Dataset name
        ann_path (str): Path to annotations
        phase (str): 'train' or 'val'
        split (int): Split number for UCF101/HMDB51
    
    Returns:
        dict: Dictionary with total frames per split
    """
    totals = {'train': 0, 'val': 0, 'test': 0}
    
    if 'sth' in dataset:
        if 'sthv1' in ann_path:
            dataset_name = 'something-something-v1'
            files_input = [osp.join(ann_path, '%s-validation.csv' % dataset_name),
                          osp.join(ann_path, '%s-train.csv' % dataset_name)]
            splits = ['val', 'train']
        else:  # sthv2
            dataset_name = 'something-something-v2'
            files_input = [osp.join(ann_path, '%s-validation.json' % dataset_name),
                          osp.join(ann_path, '%s-train.json' % dataset_name),
                          osp.join(ann_path, '%s-test.json' % dataset_name)]
            splits = ['val', 'train', 'test']
            
        for filename_input, split_name in zip(files_input, splits):
            with open(filename_input) as f:
                if filename_input.endswith('json'):
                    data = json.load(f)
                    for item in data:
                        video_id = item['id']
                        totals[split_name] += frame_counts.get(video_id, 0)
                else:  # csv
                    lines = f.readlines()
                    for line in lines:
                        items = line.rstrip().split(';')
                        video_id = items[0]
                        totals[split_name] += frame_counts.get(video_id, 0)
                        
    elif 'kinetics' in dataset:
        if '400' in ann_path:
            num_class = 400
        elif '600' in ann_path:
            num_class = 600
        elif '700' in ann_path:
            num_class = 700
            
        file_input = osp.join(ann_path, f'kinetics-{num_class}_{phase}.csv')
        with open(file_input) as f:
            lines = f.readlines()[1:]
            for line in lines:
                items = line.rstrip().split(',')
                video_id = f"{items[1]}_{int(items[2]):06d}_{int(items[3]):06d}"
                totals[phase] += frame_counts.get(video_id, 0)
                
    elif dataset == 'uffia':
        files_input = [osp.join(ann_path, f'trainlist0{split}.txt'),
                      osp.join(ann_path, f'vallist0{split}.txt')]
        splits = ['train', 'val']
        
        for filename_input, split_name in zip(files_input, splits):
            with open(filename_input) as f:
                lines = f.readlines()
                for line in lines:
                    # UFFIA format: 2022_6_13/AM_40/none/13_video_1.mp4
                    # Extract the full path without extension as video_id
                    video_path = line.rstrip()
                    video_id = video_path.replace('.mp4', '')
                    totals[split_name] += frame_counts.get(video_id, 0)
                    
    elif dataset in ['ucf101', 'hmdb51']:
        files_input = [osp.join(ann_path, f'trainlist0{split}.txt'),
                      osp.join(ann_path, f'vallist0{split}.txt')]
        splits = ['train', 'val']
        
        for filename_input, split_name in zip(files_input, splits):
            with open(filename_input) as f:
                lines = f.readlines()
                for line in lines:
                    label, name = osp.split(line.rstrip())
                    video_id = name.split('.')[0]
                    totals[split_name] += frame_counts.get(osp.join(label, video_id), 0)
    
    return totals

def main():
    """Main function"""
    args = parse_args()
    
    # Get list of videos to process
    if args.level == 1:
        src_list = glob.glob(osp.join(args.video_path, '*'))
        dest_list = [osp.join(args.out_path, osp.split(vid)[-1])
                    for vid in src_list]
    elif args.level == 2:
        src_list = glob.glob(osp.join(args.video_path, '*', '*'))
        dest_list = [osp.join(args.out_path, vid.split('/')[-2], osp.split(vid)[-1])
                    for vid in src_list]
    elif args.level == 3:
        src_list = glob.glob(osp.join(args.video_path, '*', '*', '*'))
        dest_list = [osp.join(args.out_path, vid.split('/')[-3], vid.split('/')[-2], osp.split(vid)[-1])
                    for vid in src_list]
    elif args.level == 4:
        src_list = glob.glob(osp.join(args.video_path, '*', '*', '*', '*'))
        dest_list = [osp.join(args.out_path, vid.split('/')[-4], vid.split('/')[-3], vid.split('/')[-2], osp.split(vid)[-1])
                    for vid in src_list]

    # Filter for video files only
    video_extensions = ('.mp4', '.avi', '.webm', '.mkv')
    src_list = [f for f in src_list if f.lower().endswith(video_extensions)]
    dest_list = [f for f in dest_list if f.lower().endswith(video_extensions)]

    vid_list = list(zip(src_list, dest_list))
    print(f"\nFound {len(vid_list)} videos to process")
    
    # Count frames in parallel
    pool = multiprocessing.Pool(n_thread)
    # Pass video_root for levels > 1 to generate correct relative paths
    video_root = args.video_path if args.level > 1 else None
    worker = partial(process_video, decode_type=args.lib, video_root=video_root)
    
    from tqdm import tqdm
    results = []
    for result in tqdm(pool.imap_unordered(worker, vid_list), total=len(vid_list)):
        results.append(result)
    
    pool.close()
    pool.join()
    
    # Convert results to dictionary
    frame_counts = dict(results)
    print("\nFrame count summary (first 10 videos):")
    for i, (video_id, count) in enumerate(frame_counts.items()):
        if i >= 10:  # Only show first 10 videos in summary
            break
        print(f"{video_id}: {count} frames")
    
    # Debug: Show total videos found
    print(f"\nDEBUG: Total videos in frame_counts: {len(frame_counts)}")
    if frame_counts:
        sample_key = list(frame_counts.keys())[0]
        print(f"DEBUG: Sample video_id format: '{sample_key}'")
    
    # Calculate and display total frames per split
    totals = calculate_split_totals(frame_counts, args.dataset, 
                                #  osp.join(args.ann_root, args.dataset),
                                 args.ann_root,
                                 args.phase, args.split)
    
    print("\nTotal frames per split:")
    for split, total in totals.items():
        print(f"{split}: {total:,} frames")
    
    # Generate labels
    generate_labels(args.video_path, args.out_path, frame_counts, 
                #    args.dataset, osp.join(args.ann_root, args.dataset),
                   args.dataset, args.ann_root,
                   args.level, args.phase, args.split)

if __name__ == "__main__":
    main() 