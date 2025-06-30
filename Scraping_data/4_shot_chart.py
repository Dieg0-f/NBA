import re
import time
import json
import logging
import pandas as pd
import threading
import concurrent.futures
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import tqdm

# Set up logging
log_dir = Path("Log")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Log/4_shot_chart_simple.log"),
    ]
)
logger = logging.getLogger()

# Lock per thread-safe file operations
file_lock = threading.Lock()

def setup_driver():
    """Initialize WebDriver with optimized settings"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    options.add_argument('--log-level=3')
    options.page_load_strategy = 'eager'
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def wait_for_shots(driver, timeout=15):
    """Wait for shot elements to load"""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "g.shot"))
        )
        time.sleep(1)  # Extra wait for all elements
        return True
    except:
        return False

def extract_shot_data(shot_element, gameID):
    """Extract data from a single shot element"""
    try:
        # Get title with shot information
        title_element = shot_element.find('title')
        if not title_element:
            return None
            
        title_text = title_element.text.strip()
        
        # Extract time and quarter from title (format: "Q1 - 11:27")
        time_match = re.search(r'Q(\d+)\s+-\s+(\d+:\d+)$', title_text)
        if not time_match:
            return None
            
        quarter = int(time_match.group(1))
        time_stamp = time_match.group(2)
        
        # Check if shot was made (no "MISS" in title)
        made = 0 if "MISS" in title_text else 1
        
        # Extract coordinates from path transform
        path_element = shot_element.find('path')
        if not path_element or 'transform' not in path_element.attrs:
            return None
            
        transform = path_element['transform']
        coord_match = re.search(r'translate\(([^,]+),\s*([^)]+)\)', transform)
        if not coord_match:
            return None
            
        x_coord = round(float(coord_match.group(1)))
        y_coord = round(float(coord_match.group(2)))
        
        return {
            "gameID": gameID,
            "quarter": quarter,
            "time": time_stamp,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "made": made,
            "description": title_text
        }
        
    except Exception as e:
        logger.error(f"Error extracting shot data: {str(e)}")
        return None

def determine_team_assignment(all_shots):
    """Determine which shots belong to home vs away team based on Y coordinates"""
    if not all_shots:
        return all_shots
        
    # Sort shots by Y coordinate to find the midpoint
    y_coords = [shot['y_coord'] for shot in all_shots]
    y_median = sorted(y_coords)[len(y_coords) // 2]
    
    # Assign teams based on court position
    # Typically, one team shoots "up" (lower Y) and one "down" (higher Y)
    for shot in all_shots:
        shot['is_home_team'] = shot['y_coord'] > y_median
        
    return all_shots

def append_to_csv(shots_data, csv_file):
    """Thread-safe CSV append"""
    if not shots_data:
        return False
        
    with file_lock:
        try:
            df = pd.DataFrame(shots_data)
            
            if csv_file.exists():
                df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
            else:
                df.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8')
            
            logger.info(f"Added {len(shots_data)} shots to CSV")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to CSV: {str(e)}")
            return False

def get_processed_games(csv_file):
    """Get list of already processed games"""
    if not csv_file.exists():
        return set()
    
    try:
        df = pd.read_csv(csv_file, usecols=['gameID'])
        processed = set(df['gameID'].unique())
        logger.info(f"Found {len(processed)} already processed games")
        return processed
    except Exception as e:
        logger.warning(f"Error reading existing CSV: {str(e)}")
        return set()

def process_single_game(game_link, csv_file):
    """Process one game and extract all shots"""
    gameID = game_link[25:46]
    logger.info(f"Processing game: {gameID}")
    
    url = game_link + '/game-charts?period=All'
    
    driver = setup_driver()
    try:
        # Load page
        driver.get(url)
        
        # Wait for shot elements
        if not wait_for_shots(driver):
            logger.warning(f"No shot elements loaded for {gameID}")
            return False
        
        # Parse HTML
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        shot_elements = soup.find_all('g', class_='shot')
        
        if not shot_elements:
            logger.warning(f"No shot elements found for {gameID}")
            return False
        
        logger.info(f"Found {len(shot_elements)} shot elements for {gameID}")
        
        # Extract shot data
        all_shots = []
        for shot_element in shot_elements:
            shot_data = extract_shot_data(shot_element, gameID)
            if shot_data:
                all_shots.append(shot_data)
        
        if not all_shots:
            logger.warning(f"No valid shots extracted for {gameID}")
            return False
        
        # Determine team assignments
        all_shots = determine_team_assignment(all_shots)
        
        # Save to CSV
        success = append_to_csv(all_shots, csv_file)
        
        if success:
            logger.info(f"Successfully processed {gameID}: {len(all_shots)} shots")
            return True
        else:
            logger.error(f"Failed to save shots for {gameID}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {gameID}: {str(e)}")
        return False
        
    finally:
        driver.quit()

def main():
    """Main execution function"""
    start_time = time.time()
    logger.info("Starting simplified shot chart scraping")
    
    try:
        # Setup
        csv_file = Path('Scraping_data/4_shot_chart_simple.csv')
        
        # Load game links
        with open('Scraping_data/1_game_links.json', 'r') as f:
            game_links = json.load(f)
        
        logger.info(f"Loaded {len(game_links)} total game links")
        
        # Get unprocessed games
        processed_games = get_processed_games(csv_file)
        unprocessed_games = [
            link for link in game_links 
            if link[25:46] not in processed_games
        ]
        
        total_games = len(unprocessed_games)
        logger.info(f"Found {total_games} unprocessed games")
        print(f"Processing {total_games} games out of {len(game_links)} total")
        
        if total_games == 0:
            print("No new games to process")
            return
        
        # Process games with threading
        successful = 0
        failed = 0
        max_workers = min(4, total_games)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(process_single_game, link, csv_file): link 
                for link in unprocessed_games
            }
            
            # Process results with progress bar
            with tqdm.tqdm(total=total_games, desc="Extracting Shot Charts") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    link = futures[future]
                    try:
                        success = future.result()
                        if success:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        logger.error(f"Future error for {link}: {str(e)}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful,
                        'Failed': failed,
                        'Rate': f"{(successful/(successful+failed)*100):.1f}%"
                    })
        
        # Final statistics
        execution_time = time.time() - start_time
        success_rate = (successful / total_games * 100) if total_games > 0 else 0
        
        print(f"\n=== SCRAPING COMPLETED ===")
        print(f"Successful games: {successful}")
        print(f"Failed games: {failed}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Data saved to: {csv_file}")
        
        # Show final CSV stats
        if csv_file.exists():
            try:
                final_df = pd.read_csv(csv_file)
                total_shots = len(final_df)
                unique_games = final_df['gameID'].nunique()
                made_shots = final_df['made'].sum()
                fg_percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
                
                print(f"\n=== FINAL DATA STATS ===")
                print(f"Total shots in CSV: {total_shots:,}")
                print(f"Unique games in CSV: {unique_games}")
                print(f"Overall FG%: {fg_percentage:.1f}% ({made_shots}/{total_shots})")
                
            except Exception as e:
                logger.error(f"Error reading final stats: {str(e)}")
        
        logger.info("Scraping process completed successfully")
        
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()