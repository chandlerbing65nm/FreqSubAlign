"""Generate list of train/val set for multiple video datasets.

This script generates label files for various video datasets including:
- UCF101
- HMDB51
- Kinetics (400/600/700)
- Something-Something V1/V2

The script processes video frames or flow data and creates corresponding label files
for training and validation sets.
"""

import argparse
import json
import os
import os.path as osp
import shutil
from typing import Dict, List, Tuple, Union


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset label generation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Build file label list')
    parser.add_argument('--data_path', type=str, 
                        default='/scratch/project_465001897/datasets/ucf/videos/samples',
                        help='root directory for the dataset')
    parser.add_argument('--dataset', type=str, default='ucf101', 
                        choices=['ucf101', 'kinetics400', 'sthv2'],
                        help='name of the dataset')
    parser.add_argument('--ann_root', type=str, 
                        default='/scratch/project_465001897/datasets/ucf/videos/split',
                        help='root directory for annotations')
    parser.add_argument('--out_root', type=str, 
                        default='/scratch/project_465001897/datasets/ucf/videos/split',
                        help='output directory for generated labels')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'val'],
                        help='dataset phase (train/val)')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2],
                        help='directory level for data organization')
    parser.add_argument('--source', type=str, default='flow',
                        choices=['rgb', 'flow', 'video'],
                        help='data source type')
    parser.add_argument('--split', type=int, default=1, choices=[1, 2, 3],
                        help='dataset split number')
    return parser.parse_args()


def parse_label_file(file_path: str) -> Dict[str, int]:
    """Parse annotation file to create category mapping.
    
    Args:
        file_path: Path to the label file (json/csv/txt)
        
    Returns:
        Dict mapping category names to indices
    """
    categories = []
    with open(file_path) as f:
        if file_path.endswith('json'):
            data = json.load(f)
            for i, (cat, idx) in enumerate(data.items()):
                assert i == int(idx)  # make sure the rank is right
                categories.append(cat)
        elif 'kinetics' in file_path:
            lines = f.readlines()
            categories = [c.strip().replace(' ', '_').replace(
                '"', '').replace('(', '').replace(
                ')', '').replace("'", '') for c in lines]
        else:
            # Handle UCF101/HMDB51 format (number category_name)
            lines = f.readlines()
            categories = [line.strip().split(' ')[1] for line in lines]

    if 'sthv1' in file_path:
        categories = sorted(categories)
        
    return {category: i for i, category in enumerate(categories, start=1)}


def parse_sth_video_file(file_path: str, dict_categories: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """Parse Something-Something video annotation file.
    
    Args:
        file_path: Path to the annotation file
        dict_categories: Category to index mapping
        
    Returns:
        Tuple of (video folders, category indices)
    """
    folders = []
    idx_categories = []
    
    if file_path.endswith('json'):
        with open(file_path) as f:
            data = json.load(f)
        for item in data:
            folders.append(item['id'])
            if 'test' not in file_path:
                idx_categories.append(
                    dict_categories[
                        item['template'].replace(
                            '[', '').replace(']', '')])
            else:
                idx_categories.append(0)
    elif file_path.endswith('csv'):
        with open(file_path) as f:
            lines = f.readlines()
        for line in lines:
            items = line.rstrip().split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
            
    return folders, idx_categories


def gen_sth_label(data_path: str, ann_path: str, out_path: str, source: str = 'rgb') -> None:
    """Generate labels for Something-Something datasets.
    
    Args:
        data_path: Root directory for the dataset
        ann_path: Path to annotations
        out_path: Output directory for generated labels
        source: Data source type (rgb/flow/video)
    """
    if 'sthv1' in ann_path:
        dataset_name = 'something-something-v1'
        label_file = osp.join(ann_path, f'{dataset_name}-labels.csv')
        files_input = [
            osp.join(ann_path, f'{dataset_name}-validation.csv'),
            osp.join(ann_path, f'{dataset_name}-train.csv')
        ]
        files_output = [
            osp.join(out_path, f'val_{source}.txt'),
            osp.join(out_path, f'train_{source}.txt')
        ]
    elif 'sthv2' in ann_path:
        dataset_name = 'something-something-v2'
        label_file = osp.join(ann_path, f'{dataset_name}-labels.json')
        files_input = [
            osp.join(ann_path, f'{dataset_name}-validation.json'),
            osp.join(ann_path, f'{dataset_name}-train.json'),
            osp.join(ann_path, f'{dataset_name}-test.json')
        ]
        files_output = [
            osp.join(out_path, f'val_{source}.txt'),
            osp.join(out_path, f'train_{source}.txt'),
            osp.join(out_path, f'test_{source}.txt')
        ]

    dict_categories = parse_label_file(label_file)

    for filename_input, filename_output in zip(files_input, files_output):
        folders, idx_categories = parse_sth_video_file(filename_input, dict_categories)

        output = []
        for i, (cur_folder, cur_idx) in enumerate(zip(folders, idx_categories)):
            # Count frames in each video folder
            dir_files = os.listdir(osp.join(data_path, cur_folder))
            if source == 'flow':
                dir_files = [x for x in dir_files if 'flow_x' in x]
            output.append(f'{cur_folder} {len(dir_files)} {cur_idx}')
            print(f'{i}/{len(folders)}')
            
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))


def gen_kinetics_label(data_path: str, ann_path: str, out_path: str, level: int = 1,
                      source: str = 'rgb', phase: str = 'train', start_end: bool = False) -> None:
    """Generate labels for Kinetics datasets.
    
    Args:
        data_path: Root directory for the dataset
        ann_path: Path to annotations
        out_path: Output directory for generated labels
        level: Directory level for data organization
        source: Data source type (rgb/flow/video)
        phase: Dataset phase (train/val)
        start_end: Whether to include start/end timestamps
    """
    # Determine number of classes based on dataset version
    if '400' in ann_path:
        num_class = 400
    elif '600' in ann_path:
        num_class = 600
    elif '700' in ann_path:
        num_class = 700

    label_file = osp.join(ann_path, f'kinetics-{num_class}_label_map.txt')
    file_input = osp.join(ann_path, f'kinetics-{num_class}_{phase}.csv')
    file_out = f'kinetics_{source}_{phase}.txt'

    dict_categories = parse_label_file(label_file)
    assert len(dict_categories.keys()) == num_class

    # Rename class folders if needed
    if level == 2:
        classes = os.listdir(data_path)
        for name in classes:
            dst = name.strip().replace(' ', '_').replace('"', '').replace(
                '(', '').replace(')', '').replace("'", '')
            if name != dst:
                shutil.move(
                    osp.join(data_path, name), osp.join(data_path, dst))

    count_cat = {k: 0 for k in dict_categories.keys()}
    with open(file_input) as f:
        lines = f.readlines()[1:]
        
    folders = []
    categories_list = []

    for line in lines:
        items = line.rstrip().split(',')
        folders.append(
            f"{items[1]}_{int(items[2]):06d}_{int(items[3]):06d}")
        this_category = items[0].replace(' ', '_').replace(
            '"', '').replace('(', '').replace(')', '').replace("'", '')
        categories_list.append(this_category)
        count_cat[this_category] += 1

    assert len(categories_list) == len(folders)
    missing_folders = []
    output = []

    for i, (cur_folder, category) in enumerate(zip(folders, categories_list)):
        # Handle folder names with/without start_end timestamps
        cur_folder = cur_folder if start_end else cur_folder[:11]
        cur_idx = dict_categories[category]
        
        if level == 1:
            sub_dir = cur_folder
        elif level == 2:
            sub_dir = osp.join(category, cur_folder)
            
        img_dir = osp.join(data_path, sub_dir)

        if source == 'video':
            import glob
            vid_path = glob.glob(img_dir + '*')
            if len(vid_path) == 0:
                missing_folders.append(img_dir)
            else:
                vid_name = osp.split(vid_path[0])[-1]
                output.append(f'{osp.join(category, vid_name)} {cur_idx}')
        else:
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
            else:
                dir_files = os.listdir(img_dir)
                if source == 'flow':
                    dir_files = [x for x in dir_files if 'flow_x' in x]
                output.append(f'{sub_dir} {len(dir_files)} {cur_idx}')

        print(f'{i}/{len(folders)}, missing {len(missing_folders)}')
        
    with open(osp.join(out_path, file_out), 'w') as f:
        f.write('\n'.join(output))
    with open(osp.join(out_path, 'missing_' + file_out), 'w') as f:
        f.write('\n'.join(missing_folders))


def gen_label(data_path: str, ann_path: str, out_path: str, source: str, split: int) -> None:
    """Generate labels for UCF101 and HMDB51 datasets.
    
    Args:
        data_path: Root directory for the dataset
        ann_path: Path to annotations
        out_path: Output directory for generated labels
        source: Data source type (rgb/flow/video)
        split: Dataset split number
    """
    label_file = osp.join(ann_path, 'classInd.txt')
    files_input = [
        osp.join(ann_path, f'trainlist0{split}.txt'),
        osp.join(ann_path, f'testlist0{split}.txt')
    ]
    files_output = [
        osp.join(out_path, f'train_{source}_split_{split}.txt'),
        osp.join(out_path, f'val_{source}_split_{split}.txt')
    ]

    dict_categories = parse_label_file(label_file)
    print(f"Found {len(dict_categories)} categories in {label_file}")

    for filename_input, filename_output in zip(files_input, files_output):
        print(f"\nProcessing {filename_input}")
        with open(filename_input) as f:
            lines = f.readlines()
            
        folders = []
        categories_list = []
        for line in lines:
            label, name = osp.split(line.rstrip())
            folders.append(name.split('.')[0])
            categories_list.append(label)

        output = []
        missing_videos = []
        for i, (cur_folder, category) in enumerate(zip(folders, categories_list)):
            cur_idx = dict_categories[category]
            category_dir = osp.join(data_path, category)
            
            if source == 'video':
                # Look for the video file in the category directory with case-insensitive matching
                all_files = os.listdir(category_dir)
                target_lower = cur_folder.lower()
                matching_files = [f for f in all_files if f.lower().startswith(target_lower + '.')]
                
                if not matching_files:
                    print(f"Warning: No video found for {cur_folder} in {category_dir}")
                    missing_videos.append(osp.join(category_dir, cur_folder))
                    continue
                    
                vid_name = matching_files[0]  # Use the first matching file
                output.append(f'{osp.join(category, vid_name)} {cur_idx}')
            else:  # rgb or flow
                # For rgb/flow, we need to count frames in the directory
                frame_dir = osp.join(category_dir, cur_folder)
                if not os.path.exists(frame_dir):
                    print(f"Warning: Frame directory not found: {frame_dir}")
                    missing_videos.append(frame_dir)
                    continue
                    
                dir_files = os.listdir(frame_dir)
                if source == 'flow':
                    dir_files = [x for x in dir_files if 'flow_x' in x]
                output.append(f'{osp.join(category, cur_folder)} {len(dir_files)} {cur_idx}')
                
            if i % 100 == 0:
                print(f'Processed {i}/{len(folders)} videos')
            
        print(f"\nWriting output to {filename_output}")
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
            
        if missing_videos:
            missing_file = filename_output.replace('.txt', '_missing.txt')
            print(f"\nWriting list of {len(missing_videos)} missing videos to {missing_file}")
            with open(missing_file, 'w') as f:
                f.write('\n'.join(missing_videos))


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    
    if not osp.exists(args.out_root):
        os.system(f'mkdir -p {args.out_root}')
        
    if 'sth' in args.dataset:
        gen_sth_label(args.data_path, args.ann_root, args.out_root, args.source)
    elif 'kinetics' in args.dataset:
        gen_kinetics_label(args.data_path, args.ann_root, args.out_root, args.level, args.source, args.phase)
    elif args.dataset in ['ucf101', 'hmdb51']:
        gen_label(args.data_path, args.ann_root, args.out_root, args.source, args.split)


if __name__ == "__main__":
    main()