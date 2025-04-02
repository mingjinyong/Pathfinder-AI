import os
import pandas as pd
from pathlib import Path
import shutil
import re

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def standardize_column_names(df, year):
    """Standardize column names based on the year"""
    column_mapping = {
        # Geographic columns
        'AREA': 'area',
        'AREA_TITLE': 'area_title',
        'AREA_TYPE': 'area_type',
        
        # Industry columns
        'NAICS': 'naics',
        'NAICS_TITLE': 'naics_title',
        'OWN_CODE': 'own_code',
        
        # Occupation columns
        'OCC_CODE': 'occ_code',
        'OCC_TITLE': 'occ_title',
        'O_GROUP': 'o_group',
        'I_GROUP': 'i_group',
        'occ code': 'occ_code',
        'occ title': 'occ_title',
        'group': 'o_group',
        
        # Employment columns
        'TOT_EMP': 'tot_emp',
        'EMP_PRSE': 'emp_prse',
        'JOBS_1000': 'jobs_1000',
        'jobs_1000_orig': 'jobs_1000',
        'LOC_QUOTIENT': 'loc_quotient',
        'PCT_TOTAL': 'pct_total',
        
        # Hourly wage columns
        'H_MEAN': 'h_mean',
        'H_MEDIAN': 'h_median',
        'H_PCT10': 'h_pct10',
        'H_PCT25': 'h_pct25',
        'H_PCT75': 'h_pct75',
        'H_PCT90': 'h_pct90',
        
        # Annual wage columns
        'A_MEAN': 'a_mean',
        'A_MEDIAN': 'a_median',
        'A_PCT10': 'a_pct10',
        'A_PCT25': 'a_pct25',
        'A_PCT75': 'a_pct75',
        'A_PCT90': 'a_pct90',
        
        # Flag columns
        'HOURLY': 'hourly',
        'ANNUAL': 'annual',
        
        # Quality columns
        'MEAN_PRSE': 'mean_prse'
    }
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    renamed_columns = {}
    for col in df.columns:
        if col.lower() in [k.lower() for k in column_mapping.keys()]:
            # Find the matching key (case-insensitive)
            matching_key = next(k for k in column_mapping.keys() if k.lower() == col.lower())
            renamed_columns[col] = column_mapping[matching_key]
    
    if renamed_columns:
        print(f"\nColumn name changes:")
        for old_col, new_col in renamed_columns.items():
            print(f"  {old_col} -> {new_col}")
        df = df.rename(columns=renamed_columns)
    
    return df

def get_standard_columns():
    """Return the list of standard columns in the desired order"""
    return [
        # Geographic
        'area',
        'area_title',
        'area_type',
        
        # Industry
        'naics',
        'naics_title',
        'own_code',
        
        # Occupation
        'occ_code',
        'occ_title',
        'o_group',
        'i_group',
        
        # Employment
        'tot_emp',
        'emp_prse',
        'jobs_1000',
        'loc_quotient',
        'pct_total',
        
        # Hourly Wages
        'h_mean',
        'h_median',
        'h_pct10',
        'h_pct25',
        'h_pct75',
        'h_pct90',
        
        # Annual Wages
        'a_mean',
        'a_median',
        'a_pct10',
        'a_pct25',
        'a_pct75',
        'a_pct90',
        
        # Flags
        'hourly',
        'annual',
        
        # Quality
        'mean_prse'
    ]

def ensure_standard_columns(df):
    """Ensure the dataframe has all standard columns and remove any extra columns"""
    standard_columns = get_standard_columns()
    
    # Add missing columns with appropriate default values
    for col in standard_columns:
        if col not in df.columns:
            if col in ['hourly', 'annual']:
                df[col] = 1  # Default flag value
            else:
                df[col] = 0  # Default numeric value
    
    # Remove any columns that are not in the standard set
    columns_to_drop = [col for col in df.columns if col not in standard_columns]
    if columns_to_drop:
        print(f"Removing extra columns: {', '.join(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)
    
    # Reorder columns to match standard order
    df = df[standard_columns]
    return df

def process_and_combine_data(old_data_path, new_data_path):
    """Process all CSV files from OLD DATA and combine them into a single file in NEW DATA"""
    # Create NEW DATA directory if it doesn't exist
    ensure_directory_exists(new_data_path)
    
    # List to store all dataframes
    all_dfs = []
    
    # Process each CSV file
    for file in os.listdir(old_data_path):
        if file.endswith('.csv') and not file.startswith('standardized_'):
            year_match = re.search(r'_(\d{4})', file)
            if year_match:
                year = int(year_match.group(1))
                print(f"\nProcessing data for year {year}...")
                
                input_file = os.path.join(old_data_path, file)
                
                # Read the CSV file
                df = pd.read_csv(input_file, low_memory=False)
                print(f"Number of columns: {len(df.columns)}")
                
                # Standardize column names and ensure all standard columns exist
                df = standardize_column_names(df, year)
                df = ensure_standard_columns(df)
                
                # Add year column
                df['year'] = year
                
                # Append to list
                all_dfs.append(df)
    
    if not all_dfs:
        print("No CSV files found to process and combine.")
        return
    
    # Combine all dataframes
    print("\nCombining all data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined data
    output_file = os.path.join(new_data_path, 'combined_data.csv')
    print(f"Saving combined data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully saved combined data with {len(combined_df)} rows")

def main():
    """Main function to process and combine the data files"""
    # Define paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    old_data_path = os.path.join(base_path, 'OLD DATA')
    new_data_path = base_path  # Changed to use OEWS Data directory directly
    
    print(f"Base path: {base_path}")
    print(f"Old data path: {old_data_path}")
    print(f"Output path: {new_data_path}")
    
    print("\nStarting data processing and combination...")
    print("=" * 50)
    
    process_and_combine_data(old_data_path, new_data_path)

if __name__ == '__main__':
    main() 