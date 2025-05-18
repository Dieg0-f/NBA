import requests
import os
import time
import json
import logging
import concurrent.futures
import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configurazione logging solo su file (non a terminale)
def setup_logger():
    log_dir = '/home/diego/Documents/GitHub/NBA/Log'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('video_scraper_bos')
    logger.setLevel(logging.INFO)
    
    # Formattatore con più dettagli
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
    
    # Solo handler per file, nessun handler per console
    file_handler = logging.FileHandler(os.path.join(log_dir, 'scrape_video_bos.log'), mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Rimuovi handler esistenti per evitare duplicati
    if logger.handlers:
        logger.handlers.clear()
    
    logger.addHandler(file_handler)
    
    return logger

# Inizializza il logger
logger = setup_logger()

# Funzione per creare un driver Chrome
def create_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--silent")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    return webdriver.Chrome(options=chrome_options)

# Funzione per processare un singolo video
def process_video(play_data):
    driver = create_driver()
    # Inizializza il dizionario con download = False
    updated_data = {
        'file_name': play_data['file_name'],
        'clip_link': play_data['clip_link'],
        'download': False
    }
    
    try:
        file_name = play_data['file_name']
        url = play_data['clip_link']
        
        logger.info(f"Elaborazione di {file_name} da {url}")
        
        # Controlla se il file esiste già
        output_path = f"Video_bos/{file_name}"
        if not output_path.endswith('.mp4'):
            output_path += '.mp4'
            
        if os.path.exists(output_path):
            logger.info(f"File {output_path} già esistente, lo segno come scaricato.")
            updated_data['download'] = True
            driver.quit()
            return updated_data
        
        # Accedi alla pagina
        driver.get(url)
        
        # Attesa esplicita per il tag video con timeout più breve
        try:
            video_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "video.vjs-tech"))
            )
            
            # Estrai l'URL del video
            video_url = video_element.get_attribute("src")
            
            if not video_url:
                logger.error(f"URL video non trovato per {file_name}")
                driver.quit()
                return updated_data  # download rimane False
            
            # Scarica il video
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': url
            }
            
            # Timeout più breve per le richieste
            video_response = requests.get(video_url, headers=headers, stream=True, timeout=30)
            
            # Crea directory Video_bos se non esiste
            os.makedirs("Video_bos", exist_ok=True)
            
            # Salva il video
            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verifica che il file sia stato effettivamente creato e non abbia dimensione zero
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Video salvato con successo: {output_path}")
                updated_data['download'] = True
            else:
                logger.error(f"Il file {output_path} sembra vuoto o non creato correttamente")
                
            return updated_data
            
        except TimeoutException:
            logger.error(f"Timeout nell'attesa del video per {file_name}")
            return updated_data  # download rimane False
            
    except Exception as e:
        logger.error(f"Errore durante l'elaborazione di {play_data['file_name']}: {str(e)}")
        return updated_data  # download rimane False
    finally:
        driver.quit()

def main():
    start_time = time.time()
    logger.info("Inizio elaborazione video NBA")
    
    # Assicurati che la directory Video_bos esista
    os.makedirs("Video_bos", exist_ok=True)
    
    try:
        # Carica i dati
        with open('/home/diego/Documents/GitHub/NBA/Scraping_data/play_by_play_bos.json', 'r') as f:
            df = json.load(f)
        
        logger.info(f"Trovati {len(df)} partite nel file JSON")
        
        # Raccogli tutti i play da elaborare
        all_plays = []
        for game_idx, game in enumerate(df):
            for play_idx, play in enumerate(game):
                if play.get('clip_link'):
                    data = {
                        'file_name': f"{play['gameID']}_{play['quarter']}_{play['time']}",
                        'clip_link': f"https://www.nba.com/{play['clip_link']}"
                    }
                    all_plays.append(data)
        
        logger.info(f"Totale clip video da elaborare: {len(all_plays)}")
        print(f"Totale clip video da elaborare: {len(all_plays)}")
        
        # Elaborazione parallela con ThreadPoolExecutor
        max_workers = min(8, os.cpu_count() * 2)  # Limita il numero di thread
        logger.info(f"Avvio elaborazione parallela con {max_workers} worker")
        
        results = {"successo": 0, "fallito": 0}
        processed_plays = []
        
        # Usa tqdm per mostrare la barra di progresso
        with tqdm.tqdm(total=len(all_plays), desc="Download video", unit="clip") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_play = {executor.submit(process_video, play): play for play in all_plays}
                
                for future in concurrent.futures.as_completed(future_to_play):
                    play = future_to_play[future]
                    try:
                        updated_play = future.result()
                        processed_plays.append(updated_play)
                        
                        if updated_play['download']:
                            results["successo"] += 1
                            pbar.set_postfix(successi=results["successo"], falliti=results["fallito"])
                        else:
                            results["fallito"] += 1
                            pbar.set_postfix(successi=results["successo"], falliti=results["fallito"])
                            
                    except Exception as e:
                        logger.error(f"Errore non gestito per {play['file_name']}: {str(e)}")
                        results["fallito"] += 1
                        # Aggiungi il play con download impostato a False
                        processed_plays.append({
                            'file_name': play['file_name'],
                            'clip_link': play['clip_link'],
                            'download': False
                        })
                    
                    # Aggiorna la barra di progresso
                    pbar.update(1)
        
        # Salva i risultati in un file JSON
        result_file = '/home/diego/Documents/GitHub/NBA/Scraping_data/video_bos_download_results.json'
        with open(result_file, 'w') as f:
            json.dump(processed_plays, f, indent=2)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Elaborazione completata in {elapsed_time:.2f} secondi")
        logger.info(f"Risultati: {results['successo']} video scaricati con successo, {results['fallito']} falliti")
        logger.info(f"I risultati sono stati salvati in {result_file}")
        
        # Messaggio di riepilogo a terminale
        print(f"\nElaborazione completata in {elapsed_time:.2f} secondi")
        print(f"Risultati: {results['successo']} video scaricati con successo, {results['fallito']} falliti")
        print(f"I risultati sono stati salvati in {result_file}")
        print(f"Log dettagliato disponibile in: /home/diego/Documents/GitHub/NBA/Log/scrape_video_bos.log")
        
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione principale: {str(e)}")
        print(f"Si è verificato un errore: {str(e)}")
    
    logger.info("Script terminato")

if __name__ == "__main__":
    main()