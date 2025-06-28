import re
import time
import json
import logging
import concurrent.futures
from datetime import datetime
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import tqdm

# Scarica la posizione di tutti i tiri delle partite dei boston filtrando da game_links.json

# Set up logging
log_dir = Path("Log")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Log/shot_chart_bos.log"),
    ]
)
logger = logging.getLogger()

def setup_driver():
    """Initialize and configure WebDriver"""
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def process_game(link):
    """Process a single game to extract shot data"""
    gameID = link[25:46]
    
    # Skip if not a Boston game
    if 'bos' not in gameID:
        return [], 0
    
    logger.info(f"Processing game: {gameID}")
    
    url = link + '/game-charts?period=All'
    game_shots = []
    
    driver = setup_driver()
    try:
        driver.get(url)
        # Reduced wait time with more reliable page loading check
        time.sleep(0.5)
        
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        court_elements = soup.find_all('g', class_='court')
        
        home = False
        for court in court_elements:
            shot_elements = court.find_all('g', class_='shot')
            
            for shot in shot_elements:
                try:
                    # Extract shot information
                    title_text = shot.find('title').text.strip()
                    timer = title_text[-5:]
                    quarter = int(title_text[-9:-8])
                    
                    coord_text = shot.find('path')['transform']
                    coords_match = re.search(r'translate\(([^,]+),\s*([^)]+)\)', coord_text)
                    
                    if coords_match:
                        x_coord = round(float(coords_match.group(1)))
                        y_coord = round(float(coords_match.group(2)))
                        
                        shot_data = {
                            "gameID": gameID,
                            "timer": timer,
                            "quarter": quarter,
                            "x_coord": x_coord,
                            "y_coord": y_coord,
                            "is_home_team": home
                        }
                        
                        game_shots.append(shot_data)
                except Exception as e:
                    logger.error(f"Error processing shot in game {gameID}: {str(e)}")
            
            home = not home
        
        logger.info(f"Successfully extracted {len(game_shots)} shots from game {gameID}")
        return game_shots, 1  # Return shots and count of links opened
    
    except Exception as e:
        logger.error(f"Error processing game {gameID}: {str(e)}")
        return [], 1  # Still count the link as opened even if there's an error
    
    finally:
        driver.quit()

def main():
    start_time = time.time()
    logger.info("Starting shot chart scraping process")
    
    try:
        # Load game links
        with open('Scraping_data/1_game_links.json', 'r') as f:
            game_links = json.load(f)
        
        logger.info(f"Loaded {len(game_links)} game links")
        
        # Filter Boston games only
        boston_games = [link for link in game_links if 'bos' in link[25:46]]
        total_games = len(boston_games)
        logger.info(f"Found {total_games} Boston games to process")
        
        # Process games in parallel with progress bar
        shots_data = []
        links_opened = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_game, link): link for link in boston_games}
            
            # Create progress bar
            with tqdm.tqdm(total=total_games, desc="Scraping Progress") as progress_bar:
                for future in concurrent.futures.as_completed(futures):
                    link = futures[future]
                    try:
                        game_shots, opened = future.result()
                        shots_data.extend(game_shots)
                        links_opened += opened
                    except Exception as e:
                        logger.error(f"Error with game {link}: {str(e)}")
                        links_opened += 1  # Count as opened even if exception in future handling
                    
                    # Update progress bar
                    progress_bar.update(1)

        # Save data to JSON file
        with open('Scraping_data/4_shot_chart.json', 'w') as f:
            json.dump(shots_data, f, indent=2)
        
        # Log summary info including link count
        logger.info(f"Scraping completed. Total shots: {len(shots_data)}")
        logger.info(f"Total links opened: {links_opened}")
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
        print(f"Scraping completed. Saved {len(shots_data)} shots to Scraping_data/4_shot_chart.json")
        
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()