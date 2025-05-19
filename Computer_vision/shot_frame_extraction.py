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
from tqdm import tqdm 
from ultralytics import YOLO

# Configurazione del logging
LOG_FILE = '/home/diego/Documents/GitHub/NBA/Log/shot_frame_extraction.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Verifica disponibilità CUDA e configura l'ambiente
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    logger.info(f"CUDA disponibile: {torch.cuda.get_device_name(0)}")
    # Configurazione per prestazioni ottimali su NVIDIA
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    logger.warning("CUDA non disponibile, utilizzo CPU")

# Classe per sopprimere l'output di YOLO
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

def preprocess_timer_image(timer_region):
    # Ottimizzazione: riduzione dei passaggi di conversione per minimizzare transfer CPU/GPU
    gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Utilizzo di operazioni vettorizzate con numpy per migliori prestazioni
    white_pixels = np.count_nonzero(thresh == 255)
    black_pixels = np.count_nonzero(thresh == 0)
    if white_pixels < black_pixels:
        thresh = cv2.bitwise_not(thresh)
    return thresh

def read_timer_text(timer_region, timestamp):
    processed_img = preprocess_timer_image(timer_region)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    raw_text = pytesseract.image_to_string(processed_img, config=config).strip()
    logger.debug(f"[OCR] Testo grezzo: '{raw_text}'")
    
    parts = timestamp.split(":")
    searched_mins = int(parts[0])

    if searched_mins == 0:
        raw_text = ''.join(char for char in raw_text if char.isdigit())
        if len(raw_text) >= 2:
            try:
                secs = int(raw_text[-3:-1])
                logger.debug(f"[OCR] Testo riconosciuto :ssd... '00:{secs:02d}'")
                return 0, secs
            except ValueError:
                return None
        return None
    else:
        match = re.search(r'(\d{1,2}):(\d{2})', raw_text)
        if match:
            mins = int(match.group(1))
            secs = int(match.group(2))
            logger.debug(f"[OCR] Testo riconosciuto mm:ss... {mins}:{secs:02d}")
            return mins, secs
        
        match = re.search(r'(\d{3,4})', raw_text)
        if match and len(match.group(1)) >= 3:
            digits = match.group(1)
            mins = int(digits[:-2])
            secs = int(digits[-2:])
            logger.debug(f"[OCR] Testo riconosciuto mmss... {mins}:{secs:02d}")
            return mins, secs
            
        return None

def extract_frames_by_timestamp(video_path, model, output_dir="/home/diego/Documents/GitHub/NBA/Computer_vision/Frames_of_shot"):
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

    logger.info(f"Cercando il timestamp: {standardized_target} nel video: {video_filename}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Impossibile aprire il video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ottimizzazione: lettura di batch di frame per ridurre il numero di chiamate a YOLO
    check_interval = max(1, int(fps / 2))
    
    # Buffer di frame per elaborazione in batch
    batch_size = 4  # Ottimizza in base alla memoria GPU disponibile
    if total_frames > 1000:  # Solo per video lunghi
        frames_buffer = []
        indices_buffer = []
        
        for frame_idx in range(0, total_frames, check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_buffer.append(frame)
            indices_buffer.append(frame_idx)
            
            if len(frames_buffer) >= batch_size or frame_idx + check_interval >= total_frames:
                # Processo batch di frame
                with SuppressOutput():
                    results = model(frames_buffer, conf=0.5)
                
                for i, r in enumerate(results):
                    current_frame = frames_buffer[i]
                    current_idx = indices_buffer[i]
                    
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        
                        if conf > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            timer_region = current_frame[y1:y2, x1:x2].copy()
                            
                            timer_text = read_timer_text(timer_region, target_timestamp)
                            
                            if timer_text:
                                found_mins, found_secs = timer_text
                                
                                if found_mins == searched_mins and found_secs == searched_secs:
                                    out_filename = f"{name_file}_{found_mins:02d}_{found_secs:02d}_frame{current_idx}.jpg"
                                    output_path = os.path.join(output_dir, out_filename)
                                    cv2.imwrite(output_path, current_frame)                            
                                    found_timestamp = f"{found_mins:02d}:{found_secs:02d}"
                                    found_timestamps.append(found_timestamp)
                                    logger.info(f"Trovato timestamp {found_mins:02d}:{found_secs:02d} nel frame {current_idx}")
                
                frames_buffer = []
                indices_buffer = []
    else:
        # Per video più brevi, elaborazione sequenziale ottimizzata
        for frame_idx in range(0, total_frames, check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            with SuppressOutput():
                results = model(frame, conf=0.5)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        timer_region = frame[y1:y2, x1:x2].copy()
                        
                        timer_text = read_timer_text(timer_region, target_timestamp)
                        
                        if timer_text:
                            found_mins, found_secs = timer_text
                            
                            if found_mins == searched_mins and found_secs == searched_secs:
                                out_filename = f"{name_file}_{found_mins:02d}_{found_secs:02d}_frame{frame_idx}.jpg"
                                output_path = os.path.join(output_dir, out_filename)
                                cv2.imwrite(output_path, frame)                            
                                found_timestamp = f"{found_mins:02d}:{found_secs:02d}"
                                found_timestamps.append(found_timestamp)
                                logger.info(f"Trovato timestamp {found_mins:02d}:{found_secs:02d} nel frame {frame_idx}")

    cap.release()
    
    if not found_timestamps:
        logger.warning(f"Il timestamp {standardized_target} non è stato trovato nel video {video_filename}")
    
    return found_timestamps

def main():
    output_dir = "/home/diego/Documents/GitHub/NBA/Computer_vision/Frames_of_shot"
    os.makedirs(output_dir, exist_ok=True)
    
    json_output_path = "/home/diego/Documents/GitHub/NBA/Computer_vision/frames_extraction_results.json"
    start_time_total = time.time()
    
    model_path = '/home/diego/Documents/GitHub/NBA/Computer_vision/timer_detector/weights/best.pt'
    with SuppressOutput():
        # Specifico esplicitamente l'uso di CUDA per YOLO
        model = YOLO(model_path)
        model.to(DEVICE)

    cartella_video = "/home/diego/Documents/GitHub/NBA/Video_bos"
    estensioni_video = ['.mp4', '.avi', '.mov', '.mkv']

    # Carico tutti i video in una sola operazione
    tutti_i_video = [
        os.path.join(cartella_video, f) for f in os.listdir(cartella_video)
        if os.path.isfile(os.path.join(cartella_video, f)) and any(f.lower().endswith(est) for est in estensioni_video)
    ]

    logger.info("Avvio elaborazione database")
    
    # Ottimizzazione della lettura del database
    with open('/home/diego/Documents/GitHub/NBA/database_bos.json', 'r') as f:
        lines = [json.loads(line) for line in f]

    df_bos = pd.DataFrame(lines)

    # Preparazione dell'elenco video più efficiente
    video_list = []
    video_set = set(tutti_i_video)  # Usando un set per lookup più veloce

    for index, row in df_bos.iterrows():
        logger.info(f"Elaborazione entry {index} del database")
        if row["videoID"]:
            video_path = f"/home/diego/Documents/GitHub/NBA/Video_bos/{row['videoID']}"
            if video_path in video_set:
                video_list.append(video_path)
        else:
            logger.warning(f"Entry {index}: videoID mancante")
    
    results = {}
    
    # Elaborazione dei video con progress bar 
    for video_path in tqdm(video_list, desc="Elaborazione video", unit="video"):
        video_name = os.path.basename(video_path)
        logger.info(f"Elaborazione {video_name}")
        
        # Assicuriamo che la memoria CUDA sia liberata dopo ogni video
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None
        
        found = extract_frames_by_timestamp(video_path, model, output_dir)
        results[video_name] = len(found)
    
    logger.info("Riepilogo dell'estrazione dei frame:")
    
    total_frames = sum(results.values())
    total_time = time.time() - start_time_total

    json_results = {
        "risultati_per_video": {k: v for k, v in results.items()},
        "statistiche": {
            "totale_frame": total_frames,
            "media_frame_per_video": total_frames / len(results) if results else 0,
            "numero_video_elaborati": len(results),
            "tempo_elaborazione_secondi": total_time,
            "device": DEVICE,
            "gpu_info": torch.cuda.get_device_name(0) if DEVICE == 'cuda' else "N/A"
        },
        "timestamp_esecuzione": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(json_output_path, 'w') as json_file:
        json.dump(json_results, json_file, indent=4)
    
    logger.info(f"Totale frame estratti: {total_frames}")
    logger.info(f"Media frame per video: {total_frames / len(results) if results else 0:.2f}")
    logger.info(f"Tempo totale di esecuzione: {total_time:.2f} secondi")
    logger.info(f"Risultati salvati in: {json_output_path}")
    logger.info("Elaborazione completata")

if __name__ == "__main__":
    main()