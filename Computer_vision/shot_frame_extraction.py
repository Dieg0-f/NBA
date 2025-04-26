import os
import re
import cv2
import time
import json
import numpy as np
import pandas as pd
import pytesseract
import logging
import multiprocessing
import tqdm
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configurazione del logging
LOG_FILE = '/home/diego/Documents/GitHub/NBA/Log/shot_frame_extraction.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

def preprocess_timer_image(timer_region):
    """Preelabora l'immagine del timer per migliorare l'OCR (versione migliorata)"""
    # Converti in scala di grigi
    gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)

    # Ingrandisci l'immagine
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Sfoca leggermente per ridurre il rumore
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarizzazione con Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Se necessario, inverte bianco/nero
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)
    if white_pixels < black_pixels:
        thresh = cv2.bitwise_not(thresh)

    return thresh


def read_timer_text(timer_region, timestamp):
    """Legge il testo dal timer usando OCR"""
    processed_img = preprocess_timer_image(timer_region)
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    raw_text = pytesseract.image_to_string(processed_img, config=config).strip()
    logger.debug(f"[OCR] Testo grezzo: '{raw_text}'")
    
    # Parse the target timestamp
    parts = timestamp.split(":")
    searched_mins = int(parts[0])
    searched_secs = int(parts[1])

    # Normalizza vari casi possibili
    if searched_mins == 0:
        # For timestamps like 00:XX, we're mostly concerned with seconds
        raw_text = ''.join(char for char in raw_text if char.isdigit())
        if len(raw_text) >= 2:
            # Extract just the seconds part
            try:
                secs = int(raw_text[-3:-1])
                logger.debug(f"[OCR] Testo riconosciuto :ssd... '00:{secs:02d}'")
                return 0, secs
            except ValueError:
                return None
        return None
    else:
        # Try to parse MM:SS format
        match = re.search(r'(\d{1,2}):(\d{2})', raw_text)
        if match:
            mins = int(match.group(1))
            secs = int(match.group(2))
            logger.debug(f"[OCR] Testo riconosciuto mm:ss... {mins}:{secs:02d}")
            return mins, secs
        
        # Try to parse continuous digits as minutes and seconds
        match = re.search(r'(\d{3,4})', raw_text)
        if match and len(match.group(1)) >= 3:
            digits = match.group(1)
            mins = int(digits[:-2])
            secs = int(digits[-2:])
            logger.debug(f"[OCR] Testo riconosciuto mmss... {mins}:{secs:02d}")
            return mins, secs
            
        return None


def extract_frames_by_timestamp(video_path, model_path, output_dir="/home/diego/Documents/GitHub/NBA/Computer_vision/Frames_of_shot"):
    """Estrae i frame corrispondenti ai timestamp specificati"""
    # Caricamento del modello
    model = YOLO(model_path)
    
    # Get video file name without extension for output naming
    target_timestamp = video_path[-9:-4]
    video_filename = os.path.basename(video_path)
    name_file = os.path.splitext(video_filename)[0]
    
    found_timestamps = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the target timestamp
    parts = target_timestamp.split(":")
    searched_mins, searched_secs = int(parts[0]), int(parts[1])
    standardized_target = f"{searched_mins:02d}:{searched_secs:02d}"

    logger.info(f"Cercando il timestamp: {standardized_target} nel video: {video_filename}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Impossibile aprire il video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    logger.info(f"Video: {os.path.basename(video_path)}")
    logger.info(f"Durata: {video_duration:.2f} secondi ({total_frames} frames)")
    logger.info(f"FPS: {fps}")
    
    check_interval = max(1, int(fps / 2))
    start_time = time.time()
    
    for frame_idx in range(0, total_frames, check_interval):
        if frame_idx % (check_interval * 10) == 0:
            progress = (frame_idx / total_frames) * 100
            elapsed = time.time() - start_time
            remaining = (elapsed / (frame_idx + 1)) * (total_frames - frame_idx) if frame_idx > 0 else 0
            logger.info(f"Progresso: {progress:.1f}% (tempo rimanente stimato: {remaining:.1f}s)")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
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


def process_video(video_path, model_path, output_dir):
    """Funzione per processare un singolo video, da usare con multiprocessing"""
    found = extract_frames_by_timestamp(video_path, model_path, output_dir)
    return video_path, len(found)


def main():
    # Creazione della cartella di output se non esiste
    output_dir = "/home/diego/Documents/GitHub/NBA/Computer_vision/Frames_of_shot"
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp iniziale per calcolare il tempo totale di esecuzione
    start_time_total = time.time()
    
    # Caricamento del modello (path)
    model_path = '/home/diego/Documents/GitHub/NBA/Computer_vision/timer_detector/weights/best.pt'

    # Percorso della cartella con i video
    cartella_video = "/home/diego/Documents/GitHub/NBA/Video_bos"

    # Estensioni video supportate
    estensioni_video = ['.mp4', '.avi', '.mov', '.mkv']

    # Trova tutti i file video nella cartella
    tutti_i_video = [
        os.path.join(cartella_video, f) for f in os.listdir(cartella_video)
        if os.path.isfile(os.path.join(cartella_video, f)) and any(f.lower().endswith(est) for est in estensioni_video)
    ]

    # Stampa informazioni iniziali
    print(f"🔍 Caricamento database NBA...")
    logger.info("Avvio elaborazione database")
    
    with open('/home/diego/Documents/GitHub/NBA/database_bos.json', 'r') as f:
        lines = [json.loads(line) for line in f]

    df_bos = pd.DataFrame(lines)

    # Mantieni traccia dei video da elaborare
    video_list = []
    count = 0

    print(f"🎥 Ricerca di video da elaborare...")
    for index, row in df_bos.iterrows():
        logger.info(f"Elaborazione entry {index} del database")
        if row["videoID"]:
            video_path = f"/home/diego/Documents/GitHub/NBA/Video_bos/{row['videoID']}"

            if video_path in tutti_i_video:
                video_list.append(video_path)
                count += 1
                print(f"   ✓ Video trovato: {os.path.basename(video_path)}")

            if count == 5:  # Limitato a 5 video per test
                break
        else:
            logger.warning(f"Entry {index}: videoID mancante")

    # Elaborazione parallela dei video
    num_processes = min(multiprocessing.cpu_count(), len(video_list))
    print(f"\n⚙️ Avvio elaborazione con {num_processes} processi paralleli")
    logger.info(f"Avvio elaborazione con {num_processes} processi paralleli")
    
    # Inizializza la barra di progresso
    print("\n📊 Progresso elaborazione:")
    
    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Crea elenco di futures
        futures = [executor.submit(process_video, video_path, model_path, output_dir) for video_path in video_list]
        
        # Barra di progresso per monitorare il completamento
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), 
                              desc="Elaborazione video", unit="video", ncols=100):
            video_path, num_frames = future.result()
            video_name = os.path.basename(video_path)
            results[video_name] = num_frames
    
    # Registro il numero di frame estratti per ogni video
    logger.info("Riepilogo dell'estrazione dei frame:")
    
    # Calcola statistiche complessive
    total_frames = sum(results.values())
    total_time = time.time() - start_time_total
    
    # Stampa un riepilogo formattato nel terminale
    print("\n📋 RIEPILOGO DELL'ESTRAZIONE:")
    print("-" * 60)
    print(f"{'Video':<40} | {'Frame Estratti':>10}")
    print("-" * 60)
    for video_name, num_frames in results.items():
        print(f"{video_name:<40} | {num_frames:>10}")
        logger.info(f"Video: {video_name} - Frame estratti: {num_frames}")
    
    print("-" * 60)
    print(f"{'TOTALE':<40} | {total_frames:>10}")
    print(f"{'Media per video':<40} | {total_frames / len(results) if results else 0:>10.2f}")
    print("-" * 60)
    print(f"⏱️ Tempo totale di esecuzione: {total_time:.2f} secondi")
    
    logger.info(f"Totale frame estratti: {total_frames}")
    logger.info(f"Media frame per video: {total_frames / len(results) if results else 0:.2f}")
    logger.info(f"Tempo totale di esecuzione: {total_time:.2f} secondi")
    logger.info("Elaborazione completata")


if __name__ == "__main__":
    main()