from ultralytics import YOLO
import os
import yaml

# --- Configurazione dei percorsi ---
model_path = 'yolov8x.pt'  # Usa yolov8x.pt per migliori performance
data_yaml_path = 'Computer_vision/My First Project.v2i.yolov8/data.yaml'

# --- Verifica dell'esistenza dei file ---
if not os.path.exists(model_path):
    print(f"Errore: Il file del modello '{model_path}' non esiste.")
    exit()

if not os.path.exists(data_yaml_path):
    print(f"Errore: Il file data.yaml '{data_yaml_path}' non esiste.")
    exit()

# --- Caricamento del modello ---
print(f"Caricamento del modello: {model_path}")
model = YOLO(model_path)

# --- CONFIGURAZIONE OTTIMIZZATA PER MASSIMA ACCURATEZZA ---
print("Avvio dell'addestramento ottimizzato per accuratezza...")

results = model.train(
    # === PARAMETRI BASE ===
    data=data_yaml_path,
    epochs=300,                    # Aumentato per convergenza migliore
    imgsz=1280,                   # Risoluzione più alta per dettagli migliori
    batch=8,                      # Batch size fisso più piccolo per stabilità
    
    # === OTTIMIZZATORE E LEARNING RATE ===
    optimizer='AdamW',            # AdamW spesso migliore di SGD
    lr0=0.001,                   # Learning rate iniziale più basso
    lrf=0.0001,                  # Learning rate finale più basso
    momentum=0.9,                # Momentum ottimale
    weight_decay=0.001,          # Regularizzazione per evitare overfitting
    
    # === AUGMENTATION AVANZATE ===
    augment=True,
    hsv_h=0.015,                 # Variazione Hue
    hsv_s=0.7,                   # Variazione Saturazione
    hsv_v=0.4,                   # Variazione Value
    degrees=10.0,                # Rotazione (ridotta per sport)
    translate=0.1,               # Traslazione
    scale=0.5,                   # Scaling
    shear=2.0,                   # Shear transformation
    perspective=0.0001,          # Prospettiva
    flipud=0.0,                  # No flip verticale per sport
    fliplr=0.5,                  # Flip orizzontale 50%
    mosaic=1.0,                  # Mosaic augmentation
    mixup=0.15,                  # MixUp per varietà
    copy_paste=0.3,              # Copy-paste augmentation
    
    # === EARLY STOPPING E VALIDAZIONE ===
    patience=100,                # Patience aumentata
    save_period=20,              # Salva ogni 20 epoche
    val=True,                    # Validazione attiva
    plots=True,                  # Genera grafici
    
    # === CONFIGURAZIONI AVANZATE ===
    close_mosaic=30,             # Disabilita mosaic negli ultimi 30 epoch
    amp=True,                    # Mixed precision per efficienza
    fraction=1.0,                # Usa tutto il dataset
    profile=False,               # Disabilita profiling per velocità
    freeze=0,                    # Non congelare layer (0 = tutti trainable)
    
    # === LOSS FUNCTION TUNING ===
    box=7.5,                     # Box loss weight
    cls=0.5,                     # Classification loss weight  
    dfl=1.5,                     # Distribution focal loss weight
    
    # === SALVATAGGIO ===
    device=0,                    # GPU
    save=True,
    project='Computer_vision',
    name='player_detection_optimized',
    exist_ok=True,
    
    # === CONFIGURAZIONE MULTI-SCALE ===
    rect=True,                   # Rectangular training per efficienza
    cos_lr=True,                 # Cosine learning rate scheduler
    dropout=0.1,                 # Dropout per regularizzazione
    
    # === NMS TUNING ===
    conf=0.25,                   # Confidence threshold per training
    iou=0.7,                     # IoU threshold per NMS
)

print("\n" + "="*60)
print("ADDESTRAMENTO COMPLETATO")
print("="*60)

# --- VALIDAZIONE DETTAGLIATA ---
print("\n--- VALIDAZIONE MODELLO ---")
metrics = model.val(
    data=data_yaml_path,
    imgsz=1280,                  # Stessa risoluzione del training
    conf=0.001,                  # Confidence molto bassa per validation
    iou=0.6,                     # IoU per NMS in validation
    max_det=300,                 # Massimo numero di detection
    half=True,                   # Half precision per velocità
    device=0,
    plots=True,
    save_json=True,              # Salva risultati in JSON
    save_hybrid=True,            # Salva labels in formato YOLO e COCO
)

# --- STAMPA RISULTATI DETTAGLIATI ---
print(f"\n{'='*60}")
print("RISULTATI FINALI")
print(f"{'='*60}")
print(f"Fitness Score: {metrics.fitness:.4f}")
print(f"mAP50-95:     {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
print(f"mAP50:        {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
print(f"mAP75:        {metrics.box.map75:.4f} ({metrics.box.map75*100:.2f}%)")
print(f"Precision:    {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
print(f"Recall:       {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")

# --- ANALISI PER CLASSE ---
print(f"\n{'='*60}")
print("RISULTATI PER CLASSE")
print(f"{'='*60}")
class_names = metrics.names
maps_per_class = metrics.box.maps

for i, (class_name, map_score) in enumerate(zip(class_names.values(), maps_per_class)):
    print(f"{class_name:15} | mAP50-95: {map_score:.4f} ({map_score*100:.2f}%)")

# --- SUGGERIMENTI PER ULTERIORI MIGLIORAMENTI ---
print(f"\n{'='*60}")
print("SUGGERIMENTI PER MIGLIORAMENTI")
print(f"{'='*60}")

# Identifica classi con performance basse
low_performance_classes = [(name, score) for name, score in zip(class_names.values(), maps_per_class) if score < 0.3]

if low_performance_classes:
    print("⚠️  CLASSI CON BASSE PERFORMANCE:")
    for class_name, score in low_performance_classes:
        print(f"   - {class_name}: {score:.3f} ({score*100:.1f}%)")
    print("\n💡 RACCOMANDAZIONI:")
    print("   1. Aumenta il numero di immagini per queste classi")
    print("   2. Migliora la qualità delle annotazioni")
    print("   3. Considera data augmentation specifiche")

print(f"\n📁 Modello salvato in: Computer_vision/player_detection_optimized/weights/best.pt")
print(f"📊 Grafici e log disponibili in: Computer_vision/player_detection_optimized/")

# --- CODICE PER TESTARE IL MODELLO ADDESTRATO ---
print(f"\n{'='*60}")
print("CODICE PER INFERENCE")
print(f"{'='*60}")
print("""
# Per usare il modello addestrato:
from ultralytics import YOLO
import cv2

# Carica il modello ottimizzato
model = YOLO('Computer_vision/player_detection_optimized/weights/best.pt')

# Test su singola immagine
results = model('path/to/test/image.jpg', 
                conf=0.5,      # Confidence threshold
                iou=0.7,       # IoU threshold per NMS
                imgsz=1280,    # Stessa risoluzione del training
                augment=True,  # Test Time Augmentation
                agnostic_nms=False)

# Visualizza risultati
for r in results:
    im_array = r.plot()
    cv2.imshow('Detections', im_array)
    cv2.waitKey(0)
cv2.destroyAllWindows()
""")