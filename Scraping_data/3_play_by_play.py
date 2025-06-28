import os
import re
import time
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import traceback
from tqdm import tqdm
import threading

# Create log directory if it doesn't exist
log_dir = 'Log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Log/3_play_by_play.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger()

# Lock per thread-safe file operations
file_lock = threading.Lock()

def time_to_seconds(time_str):
    """Converte il formato tempo in secondi"""
    if not isinstance(time_str, str):
        return 0
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except (ValueError, AttributeError):
        return 0

def load_data():
    """Carica i dati necessari per l'elaborazione"""
    try:
        # Caricamento dati
        with open('Scraping_data/1_game_links.json', 'r') as f:
            game_links = json.load(f)
        logging.info(f"Trovati {len(game_links)} link")

        with open('Scraping_data/2_players_database.json', 'r') as f:
            players_data = json.load(f)
        logging.info(f"Trovati {len(players_data)} players")
        
        # Ottimizzazione: creazione di un dizionario per lookup rapido dei giocatori
        players_dict = {}
        for player in players_data:
            players_dict[player['Cognome']] = player['Cognome']
        
        # Ordinare i cognomi per lunghezza per migliorare il matching
        sorted_names = sorted(players_dict.keys(), key=len, reverse=True)
        
        # Caricamento play_by_play esistenti da CSV
        processed_games = set()
        csv_file = '3_play_by_play.csv'
        
        if os.path.exists(csv_file):
            try:
                # Leggi solo la colonna gameID per identificare i giochi processati
                existing_games = pd.read_csv(csv_file, usecols=['gameID'])
                processed_games = set(existing_games['gameID'].unique())
                logging.info(f"Trovati {len(processed_games)} giochi già elaborati nel CSV")
            except Exception as e:
                logging.warning(f"Errore nel caricamento del CSV esistente: {str(e)}")
                processed_games = set()
        
        return game_links, sorted_names, processed_games, csv_file
    except Exception as e:
        logging.error(f"Errore durante il caricamento dei dati: {str(e)}")
        logging.error(traceback.format_exc())
        return [], [], set(), '3_play_by_play.csv'

@lru_cache(maxsize=1024)
def find_player_name(description, sorted_names):
    """Funzione ottimizzata per trovare i nomi dei giocatori con caching"""
    if not isinstance(description, str):
        return np.nan, ""
    for p in sorted_names:
        position = description.find(p)
        if position >= 0:
            end = position + len(p)
            return description[:end], description[end:]
    return np.nan, description

def setup_driver():
    """Configurazione ottimizzata del driver Selenium"""
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-browser-side-navigation")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-notifications")
        options.page_load_strategy = 'eager'
        
        # Silenzia i log del driver
        options.add_argument('--log-level=3')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        logging.error(f"Errore nella configurazione del driver: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def append_to_csv(df, csv_file):
    """Appende i dati al CSV in modo thread-safe"""
    with file_lock:
        try:
            # Controlla se il file esiste già
            if os.path.exists(csv_file):
                # Appendi senza header
                df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
            else:
                # Crea il file con header
                df.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8')
            
            logging.info(f"Aggiunte {len(df)} righe al CSV {csv_file}")
            return True
        except Exception as e:
            logging.error(f"Errore durante l'aggiunta al CSV: {str(e)}")
            return False

def process_game(url, sorted_names, csv_file):
    """Elabora un singolo gioco e salva immediatamente i risultati"""
    logging.info(f"Caricamento pagina: {url}")
    gameID = url[25:46]
    
    # Driver dedicato per ogni thread
    driver = setup_driver()
    if not driver:
        logging.error(f"Impossibile creare il driver per {url}")
        return False
    
    try:
        driver.get(url)
        time.sleep(2)
        
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        plays = soup.find_all('article', class_='GamePlayByPlayRow_article__asoO2')
        
        play_data = []
        t = 12
        quarter = 1
        
        # Precompilazione delle regex per migliorare le prestazioni
        ast_pattern = re.compile(r"\(([A-Za-z\s\W]+)\s(\d+)\sAST\)$")
        shot_pattern = re.compile(r"\s?(\d+')?\s?(\d+PT[a-zA-Z]?)?\s?([A-Za-z\s\W]+)\s(\((\d+)\sPT[a-zA-Z]?\))(\s+)?$")
        shot_miss_pattern = re.compile(r"\s?(\d+')?\s?(\d+PT[a-zA-Z]?)?s?([A-Za-z\s\W]+)")
        sub_pattern = re.compile(r"SUB: ([A-Za-z\s\W]+) FOR ([A-Za-z\s\W]+)")
        
        for play in plays:
            time_element = play.find('span', class_='GamePlayByPlayRow_clockElement__LfzHV')
            quarter_time = time_element.text if time_element else None
            
            if quarter_time and t < int(quarter_time[:2]):
                quarter += 1
            if quarter_time:
                t = int(quarter_time[:2])
            
            article_element = play.find('div', class_='GamePlayByPlayRow_row__2iX_w')
            is_home_team = article_element.get('data-is-home-team') == 'true' if article_element else False
            desc_element = play.find('span', class_='GamePlayByPlayRow_desc__XLzrU')
            description = desc_element.text if desc_element else None
            score_element = play.find('span', class_='GamePlayByPlayRow_scoring__Ax2hd')
            
            data = {
                'gameID': gameID,
                'time': quarter_time,
                'quarter': quarter,
                'is_home_team': is_home_team,
                'action_type': None,
                'player': None,
                'made': None,
                'shot_type': None,
                'player_points': None,
                'assist_player': None,
                'assist_count': None,
                'score': None,
                'in': None,
                'out': None,
            }
            
            if description:
                desc_lower = description.lower()
                
                if 'jump ball' in desc_lower:
                    data['action_type'] = 'Jump Ball'
                elif 'block' in desc_lower:
                    data['action_type'] = 'Block'
                elif 'steal' in desc_lower:
                    data['action_type'] = 'Steal'
                elif 'free throw' in desc_lower:
                    data['action_type'] = 'Free Throw'
                elif 'turnover' in desc_lower:
                    data['action_type'] = 'Turnover'
                elif 'foul' in desc_lower:
                    data['action_type'] = 'Foul'
                elif 'sub' in desc_lower:
                    data['action_type'] = 'SUB'
                    match = sub_pattern.search(description)
                    if match:
                        data['in'] = match.group(1).strip()
                        data['out'] = match.group(2).strip()
                elif 'timeout' in desc_lower:
                    data['action_type'] = 'Timeout'
                elif 'rebound' in desc_lower:
                    data['action_type'] = 'Rebound'
                    data['time'] = play_data[-1]['time'] if play_data else None
                elif 'violation' in desc_lower:
                    data['action_type'] = 'Violation'
                elif 'instant replay' in desc_lower:
                    data['action_type'] = 'Instant replay'
                elif score_element:
                    data['action_type'] = 'Shot'
                    data['made'] = 1
                    
                    ast_match = ast_pattern.search(description)
                    if ast_match:
                        data['assist_player'] = ast_match.group(1).strip()
                        data['assist_count'] = int(ast_match.group(2))
                        description = ast_pattern.sub('', description)
                    
                    player, desc_after = find_player_name(description, tuple(sorted_names))
                    if player is not np.nan:
                        data['player'] = player
                        shot_match = shot_pattern.search(desc_after)
                        if shot_match:
                            data['shot_type'] = shot_match.group(3).strip()
                            data['player_points'] = int(shot_match.group(5)) if shot_match.group(5) else None
                        data['score'] = score_element.text
                
                elif 'miss' in desc_lower:
                    data['action_type'] = 'Shot'
                    data['made'] = 0
                    description = description[4:] if len(description) > 4 else description
                    
                    player, desc_after = find_player_name(description, tuple(sorted_names))
                    if player is not np.nan:
                        data['player'] = player
                        shot_match = shot_miss_pattern.search(desc_after)
                        if shot_match:
                            data['shot_type'] = shot_match.group(3).strip() if shot_match.group(3) else None
            
            play_data.append(data)
        
        logging.info(f"Scraped {len(plays)} azioni per {gameID}")
        
        if not play_data:
            logging.warning(f"Nessuna azione trovata per {gameID}")
            return False
        
        # Processare direttamente il DataFrame
        df = pd.DataFrame(play_data)
        
        # Calcolo shot clock in modo sicuro
        df['seconds_from_start'] = df['time'].apply(time_to_seconds)
        df['shot_clock'] = np.nan
        
        # Inizializzazione
        if not df.empty:
            df.loc[0, 'shot_clock'] = 24.0
            
            # Elaborazione per righe successive in modo più sicuro
            for i in range(1, len(df)):
                action = df.loc[i, 'action_type']
                if pd.isna(action):
                    continue
                
                # Valori sicuri per evitare errori NoneType
                current_seconds = df.loc[i, 'seconds_from_start'] or 0
                prev_seconds = df.loc[i-1, 'seconds_from_start'] or 0
                prev_shot_clock = df.loc[i-1, 'shot_clock'] or 24.0
                
                if current_seconds > prev_seconds:
                    df.loc[i, 'shot_clock'] = 24.0 - (720 - current_seconds)
                elif action == 'Jump Ball':
                    df.loc[i, 'shot_clock'] = 24.0
                elif action in ['Block', 'Turnover', 'SUB', 'Timeout', 'Violation', 'Instant replay']:
                    delta_time = prev_seconds - current_seconds
                    df.loc[i, 'shot_clock'] = prev_shot_clock - delta_time
                elif action == 'Steal':
                    df.loc[i, 'shot_clock'] = 24.0
                elif action == 'Free Throw':
                    df.loc[i, 'shot_clock'] = prev_shot_clock
                elif action == 'Foul':
                    is_rebound_foul = (i > 1 and 
                                      df.loc[i-1, 'action_type'] == 'Rebound' and 
                                      df.loc[i-1, 'is_home_team'] == df.loc[i-2, 'is_home_team'])
                    df.loc[i, 'shot_clock'] = 14.0 if is_rebound_foul else 24.0
                elif action == 'Rebound':
                    same_team = df.loc[i-1, 'is_home_team'] == df.loc[i, 'is_home_team']
                    df.loc[i, 'shot_clock'] = 14.0 if same_team else 24.0
                elif action == 'Shot':
                    delta_time = prev_seconds - current_seconds
                    same_team = df.loc[i-1, 'is_home_team'] == df.loc[i, 'is_home_team']
                    df.loc[i, 'shot_clock'] = (prev_shot_clock - delta_time) if same_team else (24.0 - delta_time)
            
            # Tracciamento giocatori in campo in modo sicuro
            for i in range(1, 6):
                df[f'Ph{i}'] = ''
                df[f'Pa{i}'] = ''

            home_players = [None] * 5
            away_players = [None] * 5

            for idx, row in df.iterrows():
                if row['action_type'] == 'SUB':
                    if row['is_home_team']:
                        players_in_game = home_players
                        prefix = 'Ph'
                    else:
                        players_in_game = away_players
                        prefix = 'Pa'

                    if row['out'] not in players_in_game:
                        for i in range(len(players_in_game)):
                            if players_in_game[i] is None:
                                df.loc[:idx-1, f'{prefix}{i+1}'] = row['out']
                                players_in_game[i] = row['in']
                                break
                    else:
                        for i in range(len(players_in_game)):
                            if players_in_game[i] == row['out']:
                                players_in_game[i] = row['in']
                                break

                # Scrivi i valori correnti
                for i in range(5):
                    df.loc[idx, f'Ph{i+1}'] = home_players[i]
                    df.loc[idx, f'Pa{i+1}'] = away_players[i]
            
            # Salva immediatamente nel CSV
            success = append_to_csv(df, csv_file)
            if success:
                logging.info(f"Elaborazione e salvataggio completati per {gameID}")
                return True
            else:
                logging.error(f"Errore nel salvataggio per {gameID}")
                return False
        else:
            logging.warning(f"DataFrame vuoto per {gameID}")
            return False
        
    except Exception as e:
        logging.error(f"Errore durante l'elaborazione di {url}: {str(e)}")
        logging.error(traceback.format_exc())
        return False
    
    finally:
        try:
            if driver:
                driver.quit()
        except Exception as e:
            logging.error(f"Errore durante la chiusura del driver: {str(e)}")

def main():
    game_links, sorted_names, processed_games, csv_file = load_data()
    
    # Filtra i link non elaborati
    unprocessed_links = []
    for link in game_links:
        gameID = link[25:46]
        if gameID not in processed_games:
            unprocessed_links.append(f"{link}/play-by-play?period=All")
    
    logging.info(f"Da elaborare {len(unprocessed_links)} link su {len(game_links)} totali")
    print(f"Da elaborare {len(unprocessed_links)} link su {len(game_links)} totali")
    
    # Controllo se ci sono link da elaborare
    if not unprocessed_links:
        logging.info("Nessun nuovo link da elaborare")
        print("Nessun nuovo link da elaborare")
        return
    
    # Contatori per il tracking
    successful_games = 0
    failed_games = 0
    
    # Gestione delle risorse - limita il numero di thread per evitare sovraccarico
    max_workers = min(3, len(unprocessed_links))  # Ridotto per stabilità
    
    # Barra di progresso
    with tqdm(total=len(unprocessed_links), desc="Scraping partite", unit="partita") as pbar:
        # Esecuzione multithread
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(process_game, url, sorted_names, csv_file): url 
                for url in unprocessed_links
            }
            
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    success = future.result()
                    if success:
                        successful_games += 1
                    else:
                        failed_games += 1
                        
                except Exception as e:
                    failed_games += 1
                    logging.error(f"Errore durante l'elaborazione di {url}: {str(e)}")
                    logging.error(traceback.format_exc())
                
                # Aggiorna la barra di progresso
                pbar.update(1)
                pbar.set_postfix({
                    'Successo': successful_games, 
                    'Falliti': failed_games
                })
    
    # Statistiche finali
    logging.info(f"Scraping completato. Partite elaborate con successo: {successful_games}, Fallite: {failed_games}")
    print(f"Scraping completato!")
    print(f"Partite elaborate con successo: {successful_games}")
    print(f"Partite fallite: {failed_games}")
    print(f"Dati salvati in: {csv_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Errore nell'esecuzione principale: {str(e)}")
        logging.error(traceback.format_exc())