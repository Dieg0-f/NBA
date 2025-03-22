import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options


# Configura Selenium
options = Options()
options.add_argument("--headless")  # Esegui senza aprire il browser
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Avvia il WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# URL della pagina
url = "https://www.nba.com/game/bkn-vs-ind-0022401011/play-by-play?period=Q1"
driver.get(url)

# Attendi qualche secondo per il caricamento della pagina
time.sleep(2)

# Trova tutti i blocchi con le giocate
articles = driver.find_elements(By.CLASS_NAME, "GamePlayByPlayRow_article__asoO2")

data = []
for article in articles:
    try:
        # Minutaggio del tiro
        clock_span = article.find_element(By.CLASS_NAME, "GamePlayByPlayRow_clockElement__LfzHV").text.strip()

        # Link all'evento e descrizione
        a_tag = article.find_element(By.TAG_NAME, "a")
        event_text = a_tag.text.strip()
        href = a_tag.get_attribute("href")

        # Controlla se è un tiro
        if ("MISS" in event_text or "PTS" in event_text) and ("Free Throw" not in event_text):
            words = event_text.split()
            if event_text.startswith("MISS"):
                tiratore = words[1] if len(words) > 1 else ""
                esito = 0
            else:
                tiratore = words[0]
                esito = 1

            data.append({
                "link": href,
                "tiratore": tiratore,
                "canestro/miss": esito,
                "minutaggio": clock_span
            })

    except Exception as e:
        continue

# Chiudi il driver
driver.quit()

# Creazione della tabella con Pandas
df = pd.DataFrame(data)



######################
### Download video ###
######################


# Configura Selenium con Chrome
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Inizializza il driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Crea la cartella "video" se non esiste
video_folder = "video"
os.makedirs(video_folder, exist_ok=True)

for i in range(len(df)):
    video_page_url = df['link'][i]

    # Apri la pagina del video
    driver.get(video_page_url)
    time.sleep(5)  # Aspetta che la pagina carichi (puoi ottimizzare con WebDriverWait)

    try:
        # Trova il tag video (modifica il selettore in base alla struttura della pagina)
        video_element = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.TAG_NAME, "video"))
        )
        video_url = video_element.get_attribute("src")

        if video_url:
            output_filename = os.path.join(video_folder, f"{i}.mp4")

            # Scarica il video con requests
            import requests
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                with open(output_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print(f"Video salvato: {output_filename}")
            else:
                print(f"Errore nel download di {video_url} - Status Code: {response.status_code}")

        else:
            print(f"⚠️ Nessun video trovato in {video_page_url}")

    except Exception as e:
        print(f"Errore nel trovare il video in {video_page_url}: {e}")

# Chiudi il browser
driver.quit()
