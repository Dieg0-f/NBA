import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Funzione per convertire il tempo (MM:SS) in secondi totali
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except (ValueError, AttributeError):
        return 0  # In caso di errore, restituisci 0


# Configura Selenium
options = Options()
options.add_argument("--headless")  # Esegui senza aprire il browser
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Avvia il WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# URL della pagina
url = "https://www.nba.com/game/bkn-vs-ind-0022401011/game-charts?period=Q1"
driver.get(url)

# Attendi qualche secondo per il caricamento della pagina
time.sleep(2)

# Ottieni l'HTML della pagina
html_content = driver.page_source
soup = BeautifulSoup(html_content, 'html.parser')

# Inizializza la lista per memorizzare i dati
shots_data = []

# Trova tutti gli elementi "shot" nella pagina
shot_elements = soup.find_all('g', class_='shot')

# Regex per estrarre le informazioni dai titoli
player_shot_pattern = r'^(MISS )?(.*?)(\d+\'|\d+\'\s\d+\")?\s(.*?)\s\((\d+)\sPTS\)(?:\s\((.*?)\s(\d+)\sAST\))?\s(Q\d+)\s-\s(\d+:\d+)$'
miss_shot_pattern = r'^MISS\s(.*?)(\d+\'|\d+\'\s\d+\")?\s(.*?)\s(Q\d+)\s-\s(\d+:\d+)$'

for shot in shot_elements:
    # Ottieni il titolo che contiene le informazioni sul tiro
    title_text = shot.find('title').text.strip()

    # Ottieni le coordinate del tiro dal tag path
    path = shot.find('path')
    if path and 'transform' in path.attrs:
        transform_text = path['transform']
        # Estrai le coordinate dal testo transform="translate(X, Y)"
        coords_match = re.search(r'translate\(([^,]+),\s*([^)]+)\)', transform_text)
        if coords_match:
            x_coord = float(coords_match.group(1))
            y_coord = float(coords_match.group(2))
        else:
            x_coord = None
            y_coord = None
    else:
        x_coord = None
        y_coord = None

    # Analizza il titolo per estrarre le informazioni
    # Prova prima con il pattern per i tiri fatti
    match = re.search(player_shot_pattern, title_text)
    if match:
        is_miss = bool(match.group(1))
        player = match.group(2).strip()
        distance = match.group(3).strip() if match.group(3) else ""
        shot_type = match.group(4).strip()
        points = match.group(5).strip() if not is_miss else "0"
        assist_player = match.group(6).strip() if match.group(6) else ""
        assist_count = match.group(7) if match.group(7) else ""
        quarter = match.group(8).strip()
        time_remaining = match.group(9).strip()

        shots_data.append({
            'Giocatore': player,
            'Tipo di tiro': f"{distance} {shot_type}".strip(),
            'Risultato': 'Miss' if is_miss else 'Made',
            'Coordinate (x, y)': f"({x_coord}, {y_coord})",
            'Assist': f"{assist_player} ({assist_count})" if assist_player else "",
            'Tempo': time_remaining,
            'Quarto': quarter,
            'Punti': points,
            'Tempo_secondi': convert_time_to_seconds(time_remaining)  # Aggiungiamo una colonna per l'ordinamento
        })
    else:
        # Prova con il pattern per i tiri mancati
        miss_match = re.search(miss_shot_pattern, title_text)
        if miss_match:
            player = miss_match.group(1).strip()
            distance = miss_match.group(2).strip() if miss_match.group(2) else ""
            shot_type = miss_match.group(3).strip()
            quarter = miss_match.group(4).strip()
            time_remaining = miss_match.group(5).strip()

            shots_data.append({
                'Giocatore': player,
                'Tipo di tiro': f"{distance} {shot_type}".strip(),
                'Risultato': 'Miss',
                'Coordinate (x, y)': f"({x_coord}, {y_coord})",
                'Assist': "",
                'Tempo': time_remaining,
                'Quarto': quarter,
                'Punti': "0",
                'Tempo_secondi': convert_time_to_seconds(time_remaining)  # Aggiungiamo una colonna per l'ordinamento
            })
        else:
            # Se non è possibile fare match, aggiungi comunque i dati grezzi
            shots_data.append({
                'Titolo Originale': title_text,
                'Coordinate (x, y)': f"({x_coord}, {y_coord})"
            })



# Applica la funzione di conversione del tempo
for shot in shots_data:
    if 'Tempo' in shot:
        shot['Tempo_secondi'] = convert_time_to_seconds(shot['Tempo'])

# Crea un DataFrame con i dati raccolti
df = pd.DataFrame(shots_data)

# Ordina il DataFrame in base al tempo (dal tempo più alto al più basso, cioè dall'inizio alla fine del quarto)
df = df.sort_values(by=['Quarto', 'Tempo_secondi'], ascending=[True, False])
df = df.reset_index(drop=True)

# Rimuovi la colonna ausiliaria usata per l'ordinamento
if 'Tempo_secondi' in df.columns:
    df = df.drop('Tempo_secondi', axis=1)

# Salva il DataFrame in un file CSV
df.to_csv('nba_shots_data.csv', index=False)

# Imposta un limite più ampio per la larghezza della visualizzazione
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
print(df)

# Chiudi il driver
driver.quit()

print(f"Analisi completata. I dati sono stati salvati in 'nba_shots_data.csv'")