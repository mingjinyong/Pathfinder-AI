import os
import pandas as pd
import re

def combine_data_files(input_folder, output_file):
    """
    Combine all standardized CSV files from the input folder into a single file with a year column.
    
    Args:
        input_folder (str): Path to the folder containing standardized CSV files
        output_file (str): Path where the combined CSV file will be saved
    """
    # List to store all dataframes
    all_dfs = []
    
    # Process each CSV file
    for file in os.listdir(input_folder):
        if file.startswith('standardized_') and file.endswith('.csv'):
            # Extract year from filename
            year_match = re.search(r'_(\d{4})\.csv$', file)
            if year_match:
                year = int(year_match.group(1))
                print(f"Processing data for year {year}...")
                
                # Read the CSV file
                input_path = os.path.join(input_folder, file)
                df = pd.read_csv(input_path)
                
                # Add year column
                df['year'] = year
                
                # Append to list
                all_dfs.append(df)
    
    if not all_dfs:
        print("No standardized CSV files found to combine.")
        return
    
    # Combine all dataframes
    print("\nCombining all data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined data
    print(f"Saving combined data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully saved combined data with {len(combined_df)} rows")

def main():
    """Main function to combine the data files"""
    # Define paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    new_data_path = os.path.join(base_path, 'NEW DATA')
    output_file = os.path.join(base_path, 'combined_data.csv')
    
    print(f"Base path: {base_path}")
    print(f"New data path: {new_data_path}")
    print(f"Output file path: {output_file}")
    print("\nFiles in new data directory:")
    for file in os.listdir(new_data_path):
        print(f"  - {file}")
    
    print("\nStarting data combination process...")
    print("=" * 50)
    
    combine_data_files(new_data_path, output_file)

if __name__ == '__main__':
    main() 