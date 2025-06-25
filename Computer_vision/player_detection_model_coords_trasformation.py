import json
import numpy as np
import cv2
from datetime import datetime

# --- Configurazione dei Percorsi ---
INPUT_JSON_PATH = 'Computer_vision/detection_results.json'
OUTPUT_JSON_PATH = 'Computer_vision/court_coordinates_results.json'

# --- Coordinate di Riferimento del Campo da Basket (in metri) ---
COURT_POINTS = {
    'b_sx': (0, 0),           # baseline sinistra
    'b_dx': (0, 15.24),       # baseline destra  
    '3_sx': (0, 0.91),        # linea da 3 sinistra
    '3_dx': (0, 14.33),       # linea da 3 destra
    'free_bsx': (0, 5.18),    # tiro libero baseline sinistra
    'free_bdx': (0, 10.06),   # tiro libero baseline destra
    'free_asx': (5.79, 5.18), # tiro libero area sinistra
    'free_adx': (5.79, 10.06), # tiro libero area destra
    'mid_sx': (8.53, 0.91),   # metà campo sinistra
    'mid_dx': (8.53, 14.33),  # metà campo destra
    'half_sx': (14.33, 0),    # metà campo sinistra
    'half_dx': (14.33, 15.24) # metà campo destra
}

def detect_court_lines_and_calculate_homography(image_path):
    """
    Funzione per rilevare automaticamente le linee del campo e calcolare l'omografia.
    Questa è una versione semplificata - in pratica dovresti implementare 
    algoritmi più sofisticati per il rilevamento delle linee.
    """
    # Per ora utilizziamo punti di esempio - dovresti implementare il rilevamento automatico
    # Questi sono punti di esempio che dovresti identificare nell'immagine reale
    
    # Punti dell'immagine corrispondenti ai punti del campo (esempio)
    # Questi devono essere identificati automaticamente o manualmente per ogni immagine
    image_points = np.array([
        [100, 100],    # Corrisponde a b_sx (0, 0)
        [100, 600],    # Corrisponde a b_dx (0, 15.24)
        [1180, 100],   # Corrisponde a half_sx (14.33, 0)
        [1180, 600],   # Corrisponde a half_dx (14.33, 15.24)
        [640, 150],    # Corrisponde a mid_sx (8.53, 0.91)
        [640, 550],    # Corrisponde a mid_dx (8.53, 14.33)
        [300, 250],    # Corrisponde a free_asx (5.79, 5.18)
        [300, 450]     # Corrisponde a free_adx (5.79, 10.06)
    ], dtype=np.float32)
    
    # Punti corrispondenti del campo reale (in metri)
    court_points = np.array([
        [0, 0],        # b_sx
        [0, 15.24],    # b_dx
        [14.33, 0],    # half_sx
        [14.33, 15.24], # half_dx
        [8.53, 0.91],  # mid_sx
        [8.53, 14.33], # mid_dx
        [5.79, 5.18],  # free_asx
        [5.79, 10.06]  # free_adx
    ], dtype=np.float32)
    
    # Calcola la matrice di omografia
    homography_matrix, _ = cv2.findHomography(image_points, court_points, cv2.RANSAC)
    
    return homography_matrix

def transform_point_to_court(image_point, homography_matrix):
    """
    Trasforma un punto dalle coordinate dell'immagine alle coordinate del campo.
    """
    # Converti il punto in formato omogeneo
    point_homogeneous = np.array([[image_point[0], image_point[1], 1]], dtype=np.float32).T
    
    # Applica la trasformazione
    court_point_homogeneous = homography_matrix @ point_homogeneous
    
    # Normalizza per ottenere le coordinate cartesiane
    court_point = court_point_homogeneous[:2] / court_point_homogeneous[2]
    
    return court_point.flatten()

def get_detection_point(bbox, detection_type):
    """
    Calcola il punto di riferimento per la detection in base al tipo.
    - Per i punti del campo: centro del bounding box
    - Per i giocatori: media del lato inferiore del box
    """
    if detection_type in ['team', 'team1']:  # Giocatori
        # Media del lato inferiore: (x1+x2)/2, y2
        x = (bbox['bbox_xyxy']['x1'] + bbox['bbox_xyxy']['x2']) / 2
        y = bbox['bbox_xyxy']['y2']  # Lato inferiore
        return (x, y)
    else:  # Punti del campo o altri oggetti
        # Centro del bounding box
        return (bbox['bbox_center']['x'], bbox['bbox_center']['y'])

def process_detections_to_court_coordinates():
    """
    Processa il file JSON delle detection e trasforma le coordinate in coordinate del campo.
    """
    # Carica i risultati delle detection
    print(f"Caricamento dei risultati da: {INPUT_JSON_PATH}")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            detection_data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File non trovato {INPUT_JSON_PATH}")
        return
    
    # Prepara la struttura dati di output
    court_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "original_detection_file": INPUT_JSON_PATH,
            "court_coordinate_system": "meters",
            "court_dimensions": {
                "length": 14.33,  # metri
                "width": 15.24   # metri
            },
            "transformation_method": "homography",
            "total_images_processed": detection_data["metadata"]["total_images_processed"]
        },
        "court_detections": []
    }
    
    # Processa ogni immagine
    for image_data in detection_data["detections"]:
        print(f"Processando: {image_data['image_name']}")
        
        # Per questa implementazione, assumiamo che ogni immagine abbia la stessa omografia
        # In pratica, dovresti calcolare l'omografia per ogni immagine
        try:
            homography_matrix = detect_court_lines_and_calculate_homography(image_data['image_path'])
        except Exception as e:
            print(f"Errore nel calcolo dell'omografia per {image_data['image_name']}: {e}")
            # Usa una matrice identità come fallback (nessuna trasformazione)
            homography_matrix = np.eye(3, dtype=np.float32)
        
        # Struttura dati per questa immagine
        image_court_data = {
            "image_name": image_data["image_name"],
            "image_path": image_data["image_path"],
            "original_detections_count": image_data["detections_count"],
            "court_detections": []
        }
        
        # Trasforma ogni detection
        for bbox in image_data["bounding_boxes"]:
            # Calcola il punto di riferimento
            image_point = get_detection_point(bbox, bbox['class_name'])
            
            # Trasforma in coordinate del campo
            try:
                court_coordinates = transform_point_to_court(image_point, homography_matrix)
                
                # Crea l'oggetto detection con coordinate del campo
                court_detection = {
                    "detection_id": len(image_court_data["court_detections"]),
                    "class_name": bbox["class_name"],
                    "class_id": bbox["class_id"],
                    "confidence": bbox["confidence"],
                    "original_image_coordinates": {
                        "reference_point": image_point,
                        "bbox_center": (bbox["bbox_center"]["x"], bbox["bbox_center"]["y"]),
                        "bbox_bottom_center": get_detection_point(bbox, bbox['class_name'])
                    },
                    "court_coordinates": {
                        "x": float(court_coordinates[0]),  # metri
                        "y": float(court_coordinates[1]),  # metri
                    },
                    "detection_type": "player" if bbox['class_name'] in ['team', 'team1'] else "court_element"
                }
                
                image_court_data["court_detections"].append(court_detection)
                
            except Exception as e:
                print(f"Errore nella trasformazione per detection {bbox['class_name']}: {e}")
                continue
        
        court_data["court_detections"].append(image_court_data)
    
    # Salva i risultati
    print(f"Salvataggio risultati in: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(court_data, f, indent=2, ensure_ascii=False)
    
    # Statistiche finali
    total_court_detections = sum([len(img["court_detections"]) for img in court_data["court_detections"]])
    print(f"\n--- STATISTICHE TRASFORMAZIONE ---")
    print(f"Immagini processate: {len(court_data['court_detections'])}")
    print(f"Detection trasformate: {total_court_detections}")
    print(f"File salvato: {OUTPUT_JSON_PATH}")

def visualize_court_coordinates_sample():
    """
    Mostra un esempio delle coordinate trasformate per verifica.
    """
    try:
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            court_data = json.load(f)
        
        print(f"\n--- ESEMPIO COORDINATE CAMPO ---")
        
        # Mostra le prime detection della prima immagine
        if court_data["court_detections"]:
            first_image = court_data["court_detections"][0]
            print(f"Immagine: {first_image['image_name']}")
            
            for i, detection in enumerate(first_image["court_detections"][:5]):  # Prime 5
                print(f"\nDetection {i+1}:")
                print(f"  Classe: {detection['class_name']}")
                print(f"  Confidenza: {detection['confidence']:.3f}")
                print(f"  Coordinate campo: ({detection['court_coordinates']['x']:.2f}, {detection['court_coordinates']['y']:.2f}) metri")
                print(f"  Tipo: {detection['detection_type']}")
    
    except FileNotFoundError:
        print("File delle coordinate del campo non trovato. Esegui prima la trasformazione.")

if __name__ == "__main__":
    print("=== TRASFORMAZIONE COORDINATE IMMAGINE → CAMPO DA BASKET ===\n")
    
    # Esegui la trasformazione
    process_detections_to_court_coordinates()
    
    # Mostra un esempio dei risultati
    visualize_court_coordinates_sample()
    
    print(f"\n=== TRASFORMAZIONE COMPLETATA ===")
    print(f"I risultati sono stati salvati in: {OUTPUT_JSON_PATH}")
    print("\nNOTA: Questa implementazione utilizza punti di esempio per l'omografia.")
    print("Per risultati accurati, devi implementare il rilevamento automatico delle linee del campo.")