'''
Ming Jin Yong & Johnny Sheng
ACAD 222, Spring 2025
mingjiny@usc.edu & johnnysh@usc.edu
Final Project Part 1
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_standardized_data():
    """Loads the combined data from combined_data.csv"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "OEWS Data")
    file_path = os.path.join(data_dir, "combined_data.csv")
    
    try:
        print(f"\nReading file: {file_path}")  # Debug print to track file reading
        df = pd.read_csv(file_path, low_memory=False)  # Added low_memory=False to avoid dtype warnings
        print(f"Shape of data: {df.shape}")  # Debug print to see data shape
        
        # Convert tot_emp to numeric, replacing any non-numeric values with NaN
        df['tot_emp'] = pd.to_numeric(df['tot_emp'], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Current directory: {current_dir}")
        print(f"Looking for data in: {file_path}")
        raise

# Load the data
print("\nStarting data loading...")  # Debug print
data = load_standardized_data()

# Display basic information about the dataset
print("\nDataset Information:")
print("=" * 50)
print(f"Number of rows: {len(data)}")
print(f"Number of columns: {len(data.columns)}")
print("\nColumn names:", data.columns.tolist())

# Display data types and non-null counts
print("\nData Types and Non-null Counts:")
print("=" * 50)
print(data.info())

# Display missing values
print("\nMissing Values:")
print("=" * 50)
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])


def analyze_employee_growth(df):
    """
    Analyzes the growth rate of employees by job title over time
    Returns a DataFrame with job titles and their growth rates
    """
    # Create pivot table with employee counts
    pivot_table = df.pivot_table(
        values='tot_emp',
        index='occ_title',
        columns='year',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate growth rate from first to last year
    first_year = pivot_table.columns.min()
    last_year = pivot_table.columns.max()
    
    # Calculate percentage change
    growth_rates = ((pivot_table[last_year] - pivot_table[first_year]) / pivot_table[first_year] * 100)
    
    # Create a DataFrame with job titles and growth rates
    growth_df = pd.DataFrame({
        'Job Title': growth_rates.index,
        'Growth Rate (%)': growth_rates.values,
        'First Year Count': pivot_table[first_year],
        'Last Year Count': pivot_table[last_year]
    })
    
    # Sort by growth rate
    growth_df = growth_df.sort_values('Growth Rate (%)', ascending=False)
    
    # Filter out jobs with missing or invalid data
    growth_df = growth_df.dropna()
    growth_df = growth_df[growth_df['First Year Count'] > 0]  # Remove jobs that started with 0 employees
    
    return growth_df

# Create the analysis
print("\nAnalyzing employee growth by job title over time...")
growth_analysis = analyze_employee_growth(data)


# Save the analysis to a CSV file
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OEWS Data")
output_file = os.path.join(output_dir, "employee_growth_rates.csv")
growth_analysis.to_csv(output_file, index=False)
print(f"\nAnalysis saved to: {output_file}")

# Create a bar plot of the top 10 job titles by growth rate
plt.figure(figsize=(15, 8))
bars = plt.bar(range(10), growth_analysis.head(10)['Growth Rate (%)'])
plt.title('Top 10 Fastest Growing Job Titles')
plt.xlabel('Job Title')
plt.ylabel('Growth Rate (%)')
plt.xticks(range(10), growth_analysis.head(10)['Job Title'], rotation=45, ha='right')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
