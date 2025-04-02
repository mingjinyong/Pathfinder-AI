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

# Calculate and display correlations between numeric columns
print("\nCorrelation Analysis:")
print("=" * 50)
# Select only numeric columns for correlation
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_columns) > 0:
    correlation_matrix = data[numeric_columns].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Find strong correlations (absolute value > 0.7)
    strong_correlations = []
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                strong_correlations.append((numeric_columns[i], numeric_columns[j], corr))
    
    if strong_correlations:
        print("\nStrong Correlations (|r| > 0.7):")
        for col1, col2, corr in strong_correlations:
            print(f"{col1} - {col2}: {corr:.3f}")
    else:
        print("\nNo strong correlations found (|r| > 0.7)")
