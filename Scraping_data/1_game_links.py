import requests
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json

# URL della pagina
url_base = 'https://www.nba.com/games?date='

start_date = datetime(2024, 10, 22)
end_date = datetime(2025, 6, 22)
date = start_date

game_links = []

while date <= end_date:
    url = url_base + str(date)
    print(date)
    # Effettua la richiesta HTTP
    headers = {'User-Agent': 'Mozilla/5.0'}  # Per evitare problemi con alcuni siti
    response = requests.get(url, headers=headers)

    # Parsing della pagina
    soup = BeautifulSoup(response.text, 'html.parser')

    # Trova tutti i link delle partite
    for a_tag in soup.find_all('a', class_='GameCard_gcm__SKtfh'):
        link = a_tag.get('href')
        if link and link.startswith('/game/'):
            full_link = f'https://www.nba.com{link}'
            game_links.append(full_link)

    date = date + timedelta(days=1)

with open('Scraping_data/1_game_links.json', 'w') as f:
    json.dump(game_links, f, indent=2)