# Scarica tutti i nomi dei player di ogni singola partita per creare database di tutti i player

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scrape_players.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def setup_driver():
    """Configura il driver Selenium"""
    options = Options()
    options.add_argument("--headless")  # Esecuzione in background
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.page_load_strategy = 'eager'
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    return driver

def load_existing_players():
    """Carica i giocatori già salvati per evitare duplicati"""
    if os.path.exists('Scraping_data/2_players_database.json'):
        try:
            with open('Scraping_data/2_players_database.json', 'r', encoding='utf-8') as f:
                existing_players = json.load(f)
                logging.info(f"Caricati {len(existing_players)} giocatori esistenti")
                return existing_players
        except Exception as e:
            logging.warning(f"Errore nel caricamento del database esistente: {e}")
    return []

def save_players_database(players_list, format_type='both'):
    """Salva il database dei giocatori in JSON e/o CSV"""
    try:
        # Crea la directory se non esiste
        os.makedirs('Scraping_data', exist_ok=True)
        
        if format_type in ['json', 'both']:
            # Salva in JSON
            with open('Scraping_data/2_players_database.json', 'w', encoding='utf-8') as f:
                json.dump(players_list, f, indent=2, ensure_ascii=False)
            logging.info(f"Database salvato in JSON: {len(players_list)} giocatori")
        
        if format_type in ['csv', 'both']:
            # Salva in CSV
            df = pd.DataFrame(players_list)
            df.to_csv('Scraping_data/2_players_database.csv', index=False, encoding='utf-8')
            logging.info(f"Database salvato in CSV: {len(players_list)} giocatori")
            
    except Exception as e:
        logging.error(f"Errore nel salvataggio: {e}")

def extract_players_from_game(driver, game_url):
    """Estrae i giocatori da una singola partita"""
    players = []
    
    try:
        url = game_url + '/box-score'
        logging.info(f"Processando: {url}")
        
        driver.get(url)
        time.sleep(2)  # Attesa per il caricamento
        
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Trova tutti gli span con la classe dei nomi giocatori
        player_elements = soup.find_all('span', class_='GameBoxscoreTablePlayer_gbpNameFull__cf_sn')
        
        for player in player_elements:
            full_name = player.get_text(strip=True)
            if full_name:  # Controlla che il nome non sia vuoto
                parts = full_name.split()
                if len(parts) >= 2:  # Assicurati che ci siano almeno nome e cognome
                    nome = parts[0]
                    cognome = " ".join(parts[1:])
                    
                    players.append({
                        'Nome': nome,
                        'Cognome': cognome,
                        'Nome_Completo': full_name
                    })
        
        logging.info(f"Trovati {len(players)} giocatori in questa partita")
        return players
        
    except Exception as e:
        logging.error(f"Errore nell'elaborazione di {game_url}: {e}")
        return []

def main():
    """Funzione principale"""
    try:
        # Carica i link delle partite
        with open('Scraping_data/1_game_links.json', 'r') as f:
            game_links = json.load(f)
        
        logging.info(f"Trovati {len(game_links)} link delle partite")
        
        # Carica giocatori esistenti
        existing_players = load_existing_players()
        existing_names = {(p['Nome'], p['Cognome']) for p in existing_players}
        
        # Setup driver
        driver = setup_driver()
        
        all_players = []
        processed_count = 0
        
        try:
            for i, link in enumerate(game_links):
                players_in_game = extract_players_from_game(driver, link)
                
                # Aggiungi solo i nuovi giocatori
                for player in players_in_game:
                    player_key = (player['Nome'], player['Cognome'])
                    if player_key not in existing_names:
                        all_players.append(player)
                        existing_names.add(player_key)
                
                processed_count += 1
                
                # Salvataggio incrementale ogni 50 partite
                if processed_count % 50 == 0:
                    current_players = existing_players + all_players
                    save_players_database(current_players, 'json')
                    logging.info(f"Salvataggio incrementale completato ({processed_count}/{len(game_links)})")
                
                # Piccola pausa per non sovraccaricare il server
                time.sleep(1)
                
        finally:
            driver.quit()
        
        # Combina giocatori esistenti con quelli nuovi
        final_players = existing_players + all_players
        
        # Rimuovi duplicati basandosi su Nome e Cognome
        unique_players = []
        seen = set()
        
        for player in final_players:
            key = (player['Nome'], player['Cognome'])
            if key not in seen:
                unique_players.append(player)
                seen.add(key)
        
        logging.info(f"Totale giocatori unici trovati: {len(unique_players)}")
        logging.info(f"Nuovi giocatori aggiunti: {len(all_players)}")
        
        # Salvataggio finale
        save_players_database(unique_players, 'both')  # Salva sia JSON che CSV
        
        # Mostra statistiche finali
        df_final = pd.DataFrame(unique_players)
        print(f"\nStatistiche finali:")
        print(f"Totale giocatori unici: {len(unique_players)}")
        print(f"Nuovi giocatori trovati: {len(all_players)}")
        print("\nPrimi 10 giocatori:")
        print(df_final[['Nome', 'Cognome', 'Nome_Completo']].head(10))
        
    except FileNotFoundError:
        logging.error("File 'Scraping_data/1_game_links.json' non trovato!")
        print("Assicurati che il file 1_game_links.json esista nella cartella Scraping_data/")
    except Exception as e:
        logging.error(f"Errore nell'esecuzione principale: {e}")
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()