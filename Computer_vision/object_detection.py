from ultralytics import YOLO
import os

# --- Configurazione dei percorsi ---
# Assicurati che questi percorsi siano corretti e puntino ai tuoi file e directory.
# Il percorso del modello pre-addestrato
model_path = 'yolov8x.pt'

# Il percorso al tuo file data.yaml
# Questo file dovrebbe contenere i percorsi alle tue cartelle train, val e test,
# e la lista delle classi.
data_yaml_path = 'Computer_vision/My First Project.v2i.yolov8/data.yaml'

# --- Verifica dell'esistenza dei file e directory (opzionale ma consigliato) ---
if not os.path.exists(model_path):
    print(f"Errore: Il file del modello '{model_path}' non esiste.")
    print("Assicurati di aver scaricato 'yolov8x.pt' nella stessa directory dello script o specifica il percorso completo.")
    exit()

if not os.path.exists(data_yaml_path):
    print(f"Errore: Il file data.yaml '{data_yaml_path}' non esiste.")
    print("Assicurati che il percorso sia corretto.")
    exit()

# --- Caricamento del modello ---
# Carica un modello YOLOv8x pre-addestrato
print(f"Caricamento del modello: {model_path}")
model = YOLO(model_path)

# --- Addestramento del modello ---
print(f"Avvio dell'addestramento con data.yaml: {data_yaml_path}")
print("Le augmentation sono automaticamente incluse in YOLOv8 e applicate on-the-fly.")

results = model.train(
    data=data_yaml_path,
    epochs=50,             # Numero di epoche di addestramento. Aumenta per migliorare l'accuratezza.
    imgsz=640,             # Dimensione dell'immagine per l'addestramento.
    batch=-1,              # Dimensione del batch. -1 per auto batching, altrimenti un intero (es. 16, 32).
    optimizer='auto',      # Ottimizzatore. 'auto' o 'SGD', 'AdamW', ecc.
    device=0,              # Dispositivo per l'addestramento (0 per GPU 0, 'cpu' per CPU).
    augment=True,          # Abilita le augmentation integrate di YOLOv8.
    patience=50,           # Early stopping: numero di epoche senza miglioramento da attendere.
    save=True,             # Salva i pesi del modello addestrato.
    name='yolov8x_custom_detector', # Nome della directory per salvare i risultati.
    # Aggiungi altri argomenti se necessario, ad es.:
    # lr0=0.01,            # Learning rate iniziale
    # lrf=0.01,            # Learning rate finale
    # momentum=0.937,      # Momentum dello SGD
    # weight_decay=0.0005, # Decadimento del peso
    # close_mosaic=10,     # Disabilita il Mosaic negli ultimi N epoch
    # dropout=0.0,         # Dropout
)

print("\n--- Addestramento completato ---")
print(f"I risultati e i pesi del modello sono salvati nella cartella: runs/detect/yolov8x_custom_detector")

# Puoi anche validare il modello dopo l'addestramento
print("\n--- Validazione del modello ---")
metrics = model.val()  # Verranno utilizzate le impostazioni di validazione definite in data.yaml

print(f"Metriche di validazione: {metrics}")
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")

# Per testare il modello su nuove immagini:
# results = model('path/to/your/image.jpg')
# for r in results:
#    im_bgr = r.plot()  # plotto i bounding box sui risultati
#    cv2.imshow('Detection', im_bgr)
#    cv2.waitKey(0)
# cv2.destroyAllWindows()