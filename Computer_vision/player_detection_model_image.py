import os
import json
from datetime import datetime
from ultralytics import YOLO

# --- Configurazione dei Percorsi ---

# Percorso della cartella contenente le immagini da analizzare
INPUT_IMAGES_FOLDER = 'Computer_vision/Frames_of_shot'

# Percorso della cartella dove verranno salvate le immagini analizzate
OUTPUT_ANALYZED_FOLDER = 'Computer_vision/analyzed_frames'

# Percorso del modello YOLOv8x addestrato (best.pt)
TRAINED_MODEL_PATH = 'Computer_vision/player_detection_model/weights/best.pt'

# Percorso del file JSON di output per i risultati
OUTPUT_JSON_PATH = 'Computer_vision/player_detection_model_results.json'

# --- Verifica dei Percorsi ---

if not os.path.exists(INPUT_IMAGES_FOLDER):
    print(f"Errore: La cartella delle immagini da analizzare non esiste: {INPUT_IMAGES_FOLDER}")
    print("Assicurati che il percorso 'INPUT_IMAGES_FOLDER' sia corretto.")
    exit()

if not os.path.exists(TRAINED_MODEL_PATH):
    print(f"Errore: Il modello addestrato non è stato trovato al percorso specificato: {TRAINED_MODEL_PATH}")
    print("Assicurati che il percorso 'TRAINED_MODEL_PATH' sia corretto e che il file 'best.pt' esista.")
    print("Potrebbe essere necessario adattare il percorso se hai cambiato i nomi di 'project' o 'name' durante l'addestramento.")
    exit()

# --- Creazione della Cartella di Output ---
os.makedirs(OUTPUT_ANALYZED_FOLDER, exist_ok=True)
print(f"Cartella di output per le immagini analizzate: {OUTPUT_ANALYZED_FOLDER} (creata se non esistente)")

# --- Caricamento del Modello Addestrato ---
print(f"Caricamento del modello addestrato da: {TRAINED_MODEL_PATH}")
model = YOLO(TRAINED_MODEL_PATH)
print("Modello addestrato caricato con successo.")

# --- Parametri di Inferenza ---
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

print(f"\nParametri di inferenza: Confidenza={CONF_THRESHOLD}, IoU NMS={IOU_THRESHOLD}")

# --- Selezione delle prime 20 immagini ---
print(f"\nSelezione delle prime 20 immagini dalla cartella: {INPUT_IMAGES_FOLDER}")

image_files = []
# Lista le immagini nella cartella, filtrando per estensioni comuni
for f in os.listdir(INPUT_IMAGES_FOLDER):
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
        image_files.append(os.path.join(INPUT_IMAGES_FOLDER, f))

# Ordina i file per assicurarti di prendere sempre le "prime" 20 in modo consistente
image_files.sort()

# Prendi solo le prime 20 immagini
images_to_analyze = image_files[:20]

if not images_to_analyze:
    print(f"Nessuna immagine trovata nella cartella '{INPUT_IMAGES_FOLDER}' o le prime 20 non esistono.")
    exit()
else:
    print(f"Trovate {len(images_to_analyze)} immagini da analizzare (prime 20).")

# --- Esecuzione dell'Inferenza sulle Prime 20 Immagini ---
print("\nAvvio dell'analisi delle immagini selezionate...")

results = model(
    source=images_to_analyze,  # Passa la lista delle prime 20 immagini
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    save=True,
    show=False,
    project=os.path.dirname(OUTPUT_ANALYZED_FOLDER),
    name=os.path.basename(OUTPUT_ANALYZED_FOLDER),
    exist_ok=True
)

print(f"\nAnalisi completata per le prime 20 immagini. I risultati sono stati salvati in: {OUTPUT_ANALYZED_FOLDER}")

# --- Estrazione e Salvataggio dei Risultati in JSON ---
print("\nEstrazione dei dati delle detection per il file JSON...")

detection_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "model_path": TRAINED_MODEL_PATH,
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "total_images_processed": len(images_to_analyze)
    },
    "detections": []
}

# Processa ogni risultato
for i, result in enumerate(results):
    image_path = images_to_analyze[i]
    image_name = os.path.basename(image_path)
    
    # Informazioni sull'immagine
    image_info = {
        "image_name": image_name,
        "image_path": image_path,
        "image_dimensions": {
            "width": int(result.orig_shape[1]),
            "height": int(result.orig_shape[0])
        },
        "detections_count": len(result.boxes) if result.boxes is not None else 0,
        "bounding_boxes": []
    }
    
    # Estrai i bounding box se ci sono detection
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # Coordinate del bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Calcola centro, larghezza e altezza
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Informazioni del bounding box
            bbox_info = {
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox_xyxy": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                },
                "bbox_center": {
                    "x": float(center_x),
                    "y": float(center_y),
                    "width": float(width),
                    "height": float(height)
                },
                "bbox_normalized": {
                    "x1": float(x1 / result.orig_shape[1]),
                    "y1": float(y1 / result.orig_shape[0]),
                    "x2": float(x2 / result.orig_shape[1]),
                    "y2": float(y2 / result.orig_shape[0])
                }
            }
            
            image_info["bounding_boxes"].append(bbox_info)
    
    detection_data["detections"].append(image_info)

# Salva i risultati in JSON
print(f"Salvataggio dei risultati in: {OUTPUT_JSON_PATH}")
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as json_file:
    json.dump(detection_data, json_file, indent=2, ensure_ascii=False)

print(f"File JSON creato con successo: {OUTPUT_JSON_PATH}")

# --- Statistiche Finali ---
total_detections = sum([img["detections_count"] for img in detection_data["detections"]])
images_with_detections = sum([1 for img in detection_data["detections"] if img["detections_count"] > 0])

print(f"\n--- STATISTICHE FINALI ---")
print(f"Immagini processate: {len(images_to_analyze)}")
print(f"Immagini con detection: {images_with_detections}")
print(f"Totale detection: {total_detections}")
print(f"Media detection per immagine: {total_detections/len(images_to_analyze):.2f}")
print("Controlla le cartelle per vedere i risultati!")
print(f"- Immagini analizzate: {OUTPUT_ANALYZED_FOLDER}")
print(f"- Dati JSON: {OUTPUT_JSON_PATH}")