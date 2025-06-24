import os
from ultralytics import YOLO

# --- Configurazione dei Percorsi ---

# Percorso della cartella contenente le immagini da analizzare
INPUT_IMAGES_FOLDER = 'Computer_vision/Frames_of_shot'

# Percorso della cartella dove verranno salvate le immagini analizzate
OUTPUT_ANALYZED_FOLDER = 'Computer_vision/analyzed_frames'

# Percorso del modello YOLOv8x addestrato (best.pt)
TRAINED_MODEL_PATH = 'basketball_player_detection/yolov8x_fine_tuned2/weights/best.pt'

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
print("Controlla questa cartella per vedere i rilevamenti!")