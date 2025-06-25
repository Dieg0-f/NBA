import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from datetime import datetime

# --- Configurazione dei Percorsi ---
INPUT_JSON_PATH = 'Computer_vision/court_coordinates_results.json'
OUTPUT_IMAGES_FOLDER = 'Computer_vision/court_visualizations'

# --- Dimensioni del Campo da Basket (in metri) ---
COURT_LENGTH = 14.33  # metri
COURT_WIDTH = 15.24   # metri

# --- Colori per i Team ---
TEAM_COLORS = {
    'team': '#FF6B6B',    # Rosso per team 1
    'team1': '#4ECDC4',   # Turchese per team 2
    'referee': '#FFD93D', # Giallo per arbitri
    'ball': '#FF8C42',    # Arancione per palla
    'other': '#95A5A6'    # Grigio per altri elementi
}

def draw_basketball_court(ax):
    """
    Disegna un campo da basket completo con tutte le linee regolamentari.
    """
    # Colore di sfondo del campo
    ax.set_facecolor('#D2B48C')  # Colore parquet
    
    # Linee principali del campo
    court_lines = []
    
    # 1. Perimetro del campo
    court_rect = patches.Rectangle((0, 0), COURT_LENGTH, COURT_WIDTH, 
                                 linewidth=3, edgecolor='white', facecolor='none')
    ax.add_patch(court_rect)
    
    # 2. Linea di metà campo
    mid_line = plt.Line2D([COURT_LENGTH/2, COURT_LENGTH/2], [0, COURT_WIDTH], 
                         color='white', linewidth=2)
    ax.add_line(mid_line)
    
    # 3. Cerchio di metà campo
    center_circle = patches.Circle((COURT_LENGTH/2, COURT_WIDTH/2), 1.8, 
                                 linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(center_circle)
    
    # 4. Aree di tiro libero (entrambi i lati)
    # Lato sinistro
    free_throw_left = patches.Rectangle((0, 5.18), 5.79, 4.88, 
                                      linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(free_throw_left)
    
    # Lato destro
    free_throw_right = patches.Rectangle((COURT_LENGTH-5.79, 5.18), 5.79, 4.88, 
                                       linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(free_throw_right)
    
    # 5. Semicerchi di tiro libero
    # Lato sinistro
    free_circle_left = patches.Arc((5.79, COURT_WIDTH/2), 3.6, 3.6, 
                                 angle=0, theta1=-90, theta2=90, 
                                 linewidth=2, edgecolor='white')
    ax.add_patch(free_circle_left)
    
    # Lato destro
    free_circle_right = patches.Arc((COURT_LENGTH-5.79, COURT_WIDTH/2), 3.6, 3.6, 
                                  angle=0, theta1=90, theta2=270, 
                                  linewidth=2, edgecolor='white')
    ax.add_patch(free_circle_right)
    
    # 6. Linee da 3 punti (semplificate come archi)
    # Lato sinistro
    three_point_left = patches.Arc((1.5, COURT_WIDTH/2), 11.7, 11.7, 
                                 angle=0, theta1=-30, theta2=30, 
                                 linewidth=2, edgecolor='white')
    ax.add_patch(three_point_left)
    
    # Lato destro
    three_point_right = patches.Arc((COURT_LENGTH-1.5, COURT_WIDTH/2), 11.7, 11.7, 
                                  angle=0, theta1=150, theta2=210, 
                                  linewidth=2, edgecolor='white')
    ax.add_patch(three_point_right)
    
    # 7. Canestri (rappresentati come piccoli rettangoli)
    # Canestro sinistro
    basket_left = patches.Rectangle((0.15, COURT_WIDTH/2-0.23), 0.15, 0.46, 
                                  linewidth=2, edgecolor='orange', facecolor='orange')
    ax.add_patch(basket_left)
    
    # Canestro destro
    basket_right = patches.Rectangle((COURT_LENGTH-0.3, COURT_WIDTH/2-0.23), 0.15, 0.46, 
                                   linewidth=2, edgecolor='orange', facecolor='orange')
    ax.add_patch(basket_right)
    
    # Configurazione assi
    ax.set_xlim(-1, COURT_LENGTH + 1)
    ax.set_ylim(-1, COURT_WIDTH + 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Lunghezza Campo (metri)', fontsize=12)
    ax.set_ylabel('Larghezza Campo (metri)', fontsize=12)

def plot_single_frame(court_data, frame_index=0, save_path=None):
    """
    Visualizza un singolo frame con le posizioni dei giocatori.
    """
    if frame_index >= len(court_data["court_detections"]):
        print(f"Frame {frame_index} non disponibile. Massimo: {len(court_data['court_detections'])-1}")
        return None
    
    frame_data = court_data["court_detections"][frame_index]
    
    # Crea la figura
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Disegna il campo
    draw_basketball_court(ax)
    
    # Contatori per i team
    team_counts = {'team': 0, 'team1': 0, 'other': 0}
    
    # Plotta le detection
    for detection in frame_data["court_detections"]:
        x = detection["court_coordinates"]["x"]
        y = detection["court_coordinates"]["y"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        
        # Seleziona colore e dimensione
        color = TEAM_COLORS.get(class_name, TEAM_COLORS['other'])
        size = 150 if class_name in ['team', 'team1'] else 100
        
        # Plotta il punto
        scatter = ax.scatter(x, y, c=color, s=size, alpha=0.8, 
                           edgecolors='black', linewidth=1.5, zorder=5)
        
        # Aggiungi etichetta con confidenza
        ax.annotate(f'{confidence:.2f}', (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Conta i team
        if class_name in team_counts:
            team_counts[class_name] += 1
        else:
            team_counts['other'] += 1
    
    # Titolo e informazioni
    ax.set_title(f'Posizioni Giocatori - {frame_data["image_name"]}\n'
                f'Team Rosso: {team_counts["team"]} | Team Turchese: {team_counts["team1"]} | Altri: {team_counts["other"]}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Legenda
    legend_elements = []
    for team, color in TEAM_COLORS.items():
        if team in ['team', 'team1']:
            team_name = 'Team Rosso' if team == 'team' else 'Team Turchese'
            legend_elements.append(plt.scatter([], [], c=color, s=100, label=team_name))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Salva se richiesto
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Immagine salvata: {save_path}")
    
    return fig

def create_animation_frames(court_data, max_frames=10):
    """
    Crea una serie di immagini per mostrare il movimento dei giocatori nel tempo.
    """
    if not os.path.exists(OUTPUT_IMAGES_FOLDER):
        os.makedirs(OUTPUT_IMAGES_FOLDER)
    
    num_frames = min(len(court_data["court_detections"]), max_frames)
    
    print(f"Creazione di {num_frames} frame di animazione...")
    
    saved_files = []
    
    for i in range(num_frames):
        # Nome file per questo frame
        filename = f"court_frame_{i:03d}.png"
        filepath = os.path.join(OUTPUT_IMAGES_FOLDER, filename)
        
        # Crea e salva il plot
        fig = plot_single_frame(court_data, frame_index=i, save_path=filepath)
        
        if fig:
            plt.close(fig)  # Libera memoria
            saved_files.append(filepath)
    
    return saved_files

def create_heatmap_visualization(court_data):
    """
    Crea una mappa di calore delle posizioni più frequenti dei giocatori.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Disegna il campo
    draw_basketball_court(ax)
    
    # Raccogli tutte le posizioni per team
    team_positions = {'team': [], 'team1': []}
    
    for frame_data in court_data["court_detections"]:
        for detection in frame_data["court_detections"]:
            class_name = detection["class_name"]
            if class_name in team_positions:
                x = detection["court_coordinates"]["x"]
                y = detection["court_coordinates"]["y"]
                team_positions[class_name].append([x, y])
    
    # Crea heatmap per ogni team
    for team, positions in team_positions.items():
        if positions:
            positions = np.array(positions)
            
            # Crea una griglia per la heatmap
            x_grid = np.linspace(0, COURT_LENGTH, 50)
            y_grid = np.linspace(0, COURT_WIDTH, 50)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Calcola densità
            Z = np.zeros_like(X)
            for pos in positions:
                # Gaussiana centrata sulla posizione
                Z += np.exp(-((X - pos[0])**2 + (Y - pos[1])**2) / (2 * 1.0**2))
            
            # Plotta heatmap
            color = 'Reds' if team == 'team' else 'Blues'
            ax.contourf(X, Y, Z, levels=20, alpha=0.6, cmap=color)
    
    ax.set_title('Mappa di Calore - Posizioni Giocatori\n(Rosso: Team 1, Blu: Team 2)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Salva
    heatmap_path = os.path.join(OUTPUT_IMAGES_FOLDER, 'player_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap salvata: {heatmap_path}")
    
    return fig

def analyze_and_visualize_court_data():
    """
    Funzione principale che carica i dati e crea tutte le visualizzazioni.
    """
    # Carica i dati delle coordinate del campo
    print(f"Caricamento dati da: {INPUT_JSON_PATH}")
    
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            court_data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File non trovato {INPUT_JSON_PATH}")
        print("Assicurati di aver eseguito prima la trasformazione delle coordinate.")
        return
    
    # Crea cartella di output
    os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)
    
    print(f"Dati caricati: {len(court_data['court_detections'])} frame disponibili")
    
    # 1. Visualizza il primo frame
    print("\n1. Creazione visualizzazione primo frame...")
    first_frame_path = os.path.join(OUTPUT_IMAGES_FOLDER, 'first_frame_court.png')
    fig1 = plot_single_frame(court_data, frame_index=0, save_path=first_frame_path)
    if fig1:
        plt.show()
        plt.close(fig1)
    
    # 2. Crea frames di animazione
    print("\n2. Creazione frames di animazione...")
    animation_frames = create_animation_frames(court_data, max_frames=10)
    print(f"Creati {len(animation_frames)} frame di animazione")
    
    # 3. Crea heatmap
    print("\n3. Creazione mappa di calore...")
    fig3 = create_heatmap_visualization(court_data)
    plt.show()
    plt.close(fig3)
    
    # Statistiche finali
    total_detections = sum([len(frame["court_detections"]) for frame in court_data["court_detections"]])
    print(f"\n--- STATISTICHE VISUALIZZAZIONE ---")
    print(f"Frame visualizzati: {len(court_data['court_detections'])}")
    print(f"Detection totali: {total_detections}")
    print(f"Immagini create in: {OUTPUT_IMAGES_FOLDER}")
    
    return court_data

if __name__ == "__main__":
    print("=== VISUALIZZAZIONE 2D CAMPO DA BASKET ===\n")
    
    # Esegui l'analisi e visualizzazione
    court_data = analyze_and_visualize_court_data()
    
    if court_data:
        print(f"\n=== VISUALIZZAZIONE COMPLETATA ===")
        print(f"Controlla la cartella: {OUTPUT_IMAGES_FOLDER}")
        print("\nFile creati:")
        print("- first_frame_court.png (primo frame)")
        print("- court_frame_XXX.png (frames animazione)")
        print("- player_heatmap.png (mappa di calore)")