import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def convert_webm_to_mp4(args):
    webm_path, mp4_path = args
    # Read the webm video
    cap = cv2.VideoCapture(webm_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
    
    # Read and write frame by frame with progress bar
    with tqdm(total=total_frames, desc=f"Converting {os.path.basename(webm_path)}", leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            pbar.update(1)
    
    # Release everything
    cap.release()
    out.release()
    return mp4_path

# Example usage:
input_dir = '/scratch/project_465001897/datasets/ss2/videos/samples'  # your input directory
output_dir = '/scratch/project_465001897/datasets/ss2/videos/samples_mp4'  # your output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of webm files and prepare arguments for multiprocessing
webm_files = [f for f in os.listdir(input_dir) if f.endswith('.webm')]
conversion_args = [
    (
        os.path.join(input_dir, filename),
        os.path.join(output_dir, filename.replace('.webm', '.mp4'))
    )
    for filename in webm_files
]

# Determine number of processes (use 75% of available CPU cores)
n_processes = max(1, int(cpu_count() * 0.75))

# Convert videos in parallel
print(f"Starting conversion with {n_processes} processes...")
with Pool(processes=n_processes) as pool:
    list(tqdm(
        pool.imap(convert_webm_to_mp4, conversion_args),
        total=len(conversion_args),
        desc="Overall progress"
    ))