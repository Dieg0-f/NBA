import os
import re
import cv2
import time
import json
import numpy as np
import pandas as pd
import pytesseract
import logging
import sys
import torch
import concurrent.futures
from tqdm import tqdm
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict, Any

# Configure logging
LOG_FILE = '/home/diego/Documents/GitHub/NBA/Log/shot_frame_extraction.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Add console handler for better monitoring
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Class to suppress output from YOLO
class SuppressOutput:
    def __init__(self):
        self.original_stdout = sys.stdout
        self.null_output = open(os.devnull, 'w')
    
    def __enter__(self):
        sys.stdout = self.null_output
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.original_stdout
        self.null_output.close()

def preprocess_timer_image(timer_region: np.ndarray) -> np.ndarray:
    """
    Preprocess the timer region image for better OCR results.
    
    Args:
        timer_region: The cropped timer region from the original frame
        
    Returns:
        Preprocessed image ready for OCR
    """
    # Convert to grayscale if not already
    if len(timer_region.shape) == 3:
        gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = timer_region.copy()
    
    # Resize for better OCR
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding which works better for varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Count white and black pixels to determine if inversion is needed
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)
    
    # Invert if needed (text should be black on white)
    if white_pixels < black_pixels:
        thresh = cv2.bitwise_not(thresh)
    
    # Additional morphological operations for cleaner text
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

def read_timer_text(timer_region: np.ndarray, timestamp: str) -> Optional[Tuple[int, int]]:
    """
    Extract timer text from the preprocessed image.
    
    Args:
        timer_region: The cropped timer region from the original frame
        timestamp: Target timestamp to look for
        
    Returns:
        Tuple of (minutes, seconds) if found, None otherwise
    """
    processed_img = preprocess_timer_image(timer_region)
    
    # Multiple PSM modes for better recognition
    configs = [
        '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:', 
        '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:',
        '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'
    ]
    
    parts = timestamp.split(":")
    searched_mins = int(parts[0])
    
    # Try different OCR configs
    for config in configs:
        raw_text = pytesseract.image_to_string(processed_img, config=config).strip()
        logger.debug(f"[OCR] Raw text: '{raw_text}'")
        
        if searched_mins == 0:
            # Handle special case for 0 minutes
            raw_text = ''.join(char for char in raw_text if char.isdigit())
            if len(raw_text) >= 2:
                try:
                    secs = int(raw_text[-3:-1])
                    logger.debug(f"[OCR] Recognized text :ssd... '00:{secs:02d}'")
                    return 0, secs
                except ValueError:
                    continue
        else:
            # Try to match mm:ss format
            match = re.search(r'(\d{1,2}):(\d{2})', raw_text)
            if match:
                mins = int(match.group(1))
                secs = int(match.group(2))
                logger.debug(f"[OCR] Recognized text mm:ss... {mins}:{secs:02d}")
                return mins, secs
            
            # Try to match mmss format (no colon)
            match = re.search(r'(\d{3,4})', raw_text)
            if match and len(match.group(1)) >= 3:
                digits = match.group(1)
                mins = int(digits[:-2])
                secs = int(digits[-2:])
                logger.debug(f"[OCR] Recognized text mmss... {mins}:{secs:02d}")
                return mins, secs
    
    return None

def process_video_batch(frames: List[np.ndarray], model: YOLO, frame_indices: List[int], 
                       target_timestamp: str, name_file: str, output_dir: str) -> List[str]:
    """
    Process a batch of frames with the model.
    
    Args:
        frames: List of frames to process
        model: YOLO model for timer detection
        frame_indices: Original indices of the frames
        target_timestamp: Target timestamp to look for
        name_file: Base name for output files
        output_dir: Directory to save extracted frames
        
    Returns:
        List of found timestamps
    """
    found_timestamps = []
    
    # Parse the searched timestamp
    parts = target_timestamp.split(":")
    searched_mins, searched_secs = int(parts[0]), int(parts[1])
    
    # Adjust searched timestamp to match the next second (to account for slight delays)
    if searched_secs < 59:
        searched_secs += 1
    else:
        searched_mins += 1
        searched_secs = 0
    
    # Run detection on batch
    with SuppressOutput():
        results = model(frames, conf=0.5, batch=len(frames))
    
    for i, r in enumerate(results):
        frame = frames[i]
        frame_idx = frame_indices[i]
        
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 >= frame.shape[1]: x2 = frame.shape[1] - 1
                if y2 >= frame.shape[0]: y2 = frame.shape[0] - 1
                
                # Skip if the box is invalid
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                timer_region = frame[y1:y2, x1:x2].copy()
                
                # Skip if timer region is too small
                if timer_region.size == 0 or timer_region.shape[0] < 10 or timer_region.shape[1] < 10:
                    continue
                
                timer_text = read_timer_text(timer_region, target_timestamp)
                
                if timer_text:
                    found_mins, found_secs = timer_text
                    
                    if found_mins == searched_mins and found_secs == searched_secs:
                        out_filename = f"{name_file}_{found_mins:02d}_{found_secs:02d}_frame{frame_idx}.jpg"
                        output_path = os.path.join(output_dir, out_filename)
                        cv2.imwrite(output_path, frame)                            
                        found_timestamp = f"{found_mins:02d}:{found_secs:02d}"
                        found_timestamps.append(found_timestamp)
                        logger.info(f"Found timestamp {found_mins:02d}:{found_secs:02d} in frame {frame_idx}")
    
    return found_timestamps

def extract_frames_by_timestamp(video_path: str, model: YOLO, 
                               output_dir: str = "/home/diego/Documents/GitHub/NBA/Computer_vision/Frames_of_shot") -> List[str]:
    """
    Extract frames from a video that match a specific timestamp.
    
    Args:
        video_path: Path to the video file
        model: YOLO model for timer detection
        output_dir: Directory to save extracted frames
        
    Returns:
        List of found timestamps
    """
    target_timestamp = video_path[-9:-4]
    video_filename = os.path.basename(video_path)
    name_file = os.path.splitext(video_filename)[0]
    
    found_timestamps = []
    os.makedirs(output_dir, exist_ok=True)
    
    parts = target_timestamp.split(":")
    searched_mins, searched_secs = int(parts[0]), int(parts[1])
    if searched_secs < 59:
        searched_secs += 1
    else:
        searched_mins += 1
        searched_secs = 0
    standardized_target = f"{searched_mins:02d}:{searched_secs:02d}"
    logger.info(f"Looking for timestamp: {standardized_target} in video: {video_filename}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check every half second
    check_interval = max(1, int(fps / 2))
    
    # Process frames in batches to utilize GPU better
    batch_size = 8  # Adjust based on your GPU memory
    
    for start_idx in range(0, total_frames, check_interval * batch_size):
        end_idx = min(start_idx + check_interval * batch_size, total_frames)
        batch_frames = []
        batch_indices = []
        
        for frame_idx in range(start_idx, end_idx, check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
        
        if batch_frames:
            found = process_video_batch(
                batch_frames, model, batch_indices, target_timestamp, 
                name_file, output_dir
            )
            found_timestamps.extend(found)
    
    cap.release()
    
    if not found_timestamps:
        logger.warning(f"Timestamp {standardized_target} not found in video {video_filename}")
    
    return found_timestamps

def process_video(video_path: str, model: YOLO, output_dir: str) -> Tuple[str, int]:
    """
    Process a single video file.
    
    Args:
        video_path: Path to the video file
        model: YOLO model for timer detection
        output_dir: Directory to save extracted frames
        
    Returns:
        Tuple of (video_name, count of found frames)
    """
    video_name = os.path.basename(video_path)
    logger.info(f"Processing {video_name}")
    found = extract_frames_by_timestamp(video_path, model, output_dir)
    return video_name, len(found)

def main():
    # Set up directories
    output_dir = "/home/diego/Documents/GitHub/NBA/Computer_vision/Frames_of_shot"
    os.makedirs(output_dir, exist_ok=True)
    
    json_output_path = "/home/diego/Documents/GitHub/NBA/Computer_vision/frames_extraction_results.json"
    start_time_total = time.time()
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Set GPU memory usage to increase batch size
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    
    # Load YOLO model
    model_path = '/home/diego/Documents/GitHub/NBA/Computer_vision/timer_detector/weights/best.pt'
    logger.info(f"Loading model from {model_path}")
    with SuppressOutput():
        model = YOLO(model_path)
        model.to(device)  # Move model to GPU
    
    # Get list of video files
    cartella_video = "/home/diego/Documents/GitHub/NBA/Video_bos"
    estensioni_video = ['.mp4', '.avi', '.mov', '.mkv']
    tutti_i_video = [
        os.path.join(cartella_video, f) for f in os.listdir(cartella_video)
        if os.path.isfile(os.path.join(cartella_video, f)) and any(f.lower().endswith(est) for est in estensioni_video)
    ]
    logger.info("Starting database processing")
    
    # Load database
    with open('/home/diego/Documents/GitHub/NBA/database_bos.json', 'r') as f:
        lines = [json.loads(line) for line in f]
    df_bos = pd.DataFrame(lines)
    
    # Get list of videos to process
    video_list = []
    for index, row in df_bos.iterrows():
        if row["videoID"]:
            video_path = f"/home/diego/Documents/GitHub/NBA/Video_bos/{row['videoID']}"
            if video_path in tutti_i_video:
                video_list.append(video_path)
        else:
            logger.warning(f"Entry {index}: missing videoID")
    
    # Process videos in parallel
    max_workers = min(os.cpu_count(), 4)  # Limit to 4 parallel processes to avoid GPU overload
    logger.info(f"Starting parallel processing of {len(video_list)} videos with {max_workers} workers")
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dict of futures to video paths
        future_to_video = {
            executor.submit(process_video, video, model, output_dir): video 
            for video in video_list
        }
        
        # Process as completed with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_video), 
                          total=len(future_to_video), desc="Processing videos", unit="video"):
            video_path = future_to_video[future]
            try:
                video_name, count = future.result()
                results[video_name] = count
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(video_path)}: {str(e)}")
                results[os.path.basename(video_path)] = 0
    
    # Generate summary
    logger.info("Frame extraction summary:")
    
    total_frames = sum(results.values())
    total_time = time.time() - start_time_total
    
    json_results = {
        "results_by_video": {k: v for k, v in results.items()},
        "statistics": {
            "total_frames": total_frames,
            "average_frames_per_video": total_frames / len(results) if results else 0,
            "number_of_videos_processed": len(results),
            "processing_time_seconds": total_time,
            "processing_time_formatted": time.strftime("%H:%M:%S", time.gmtime(total_time)),
            "frames_per_second": total_frames / total_time if total_time > 0 else 0
        },
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_info": {
            "processor": "RTX 5060 Ti" if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }
    
    with open(json_output_path, 'w') as json_file:
        json.dump(json_results, json_file, indent=4)
    
    logger.info(f"Total extracted frames: {total_frames}")
    logger.info(f"Average frames per video: {total_frames / len(results) if results else 0:.2f}")
    logger.info(f"Total execution time: {total_time:.2f} seconds ({time.strftime('%H:%M:%S', time.gmtime(total_time))})")
    logger.info(f"Processing speed: {total_frames / total_time if total_time > 0 else 0:.2f} frames/second")
    logger.info(f"Results saved to: {json_output_path}")
    logger.info("Processing completed")

if __name__ == "__main__":
    main()