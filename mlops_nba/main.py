from mlops_nba.data_integration.data_integration import fetch_nba_team_data, integrate_data
from mlops_nba.data_analysis.data_analysis import young_potential_stars_analysis
from mlops_nba.data_processing.data_processing import process_data as process_data
import pandas as pd

from mlops_nba.config import RAW_DATA_DIR, CURATED_DATA_DIR

def main():
    # Fetch new data (example for the Celtics team)
    new_games_data = fetch_nba_team_data('BOS')
    
    # Load existing data
    existing_data = pd.read_csv('/Users/alina/vs-project/nba-mlops/mlops-nba/data/2023-2024 NBA Player Stats - Regular.csv', encoding='Windows-1252')
  # Replace with the actual file name
    
    # Integrate new data with existing data
    integrated_data = integrate_data(existing_data, new_games_data)
    
    # Process data
    processed_data = process_data(integrated_data)  # Make sure process_data is defined or imported correctly
    
    # Analyze data to identify young potential stars
    young_stars = young_potential_stars_analysis(processed_data)  # Make sure young_potential_stars_analysis is defined or imported correctly
    
    # Save the results or further process them as needed
    young_stars.to_csv(CURATED_DATA_DIR / 'young_stars.csv', index=False)

if __name__ == "__main__":
    main()
