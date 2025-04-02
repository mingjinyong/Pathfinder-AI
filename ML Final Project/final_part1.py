"""
Ming Jin Yong & Johnny Sheng
ACAD 222, Spring 2025
mingjiny@usc.edu & johnnysh@usc.edu
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def load_standardized_data() -> pd.DataFrame:
    """
    Load and preprocess the combined OEWS dataset.
    
    Returns:
        pd.DataFrame: Processed dataset with standardized columns
        
    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If there are issues with data conversion
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir / "OEWS Data"
    file_path = data_dir / "combined_data.csv"
    
    try:
        print(f"Looking for data file at: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        df['tot_emp'] = pd.to_numeric(df['tot_emp'], errors='coerce')
        if 'a_mean' in df.columns:
            df['a_mean'] = pd.to_numeric(df['a_mean'].str.replace(',', ''), errors='coerce')
        print(f"Successfully loaded data with shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"\nERROR: Data file not found at: {file_path}")
        print("Please ensure the combined_data.csv file exists in the OEWS Data directory")
        raise
    except Exception as e:
        print(f"\nERROR: Failed to process data: {str(e)}")
        raise

def analyze_employee_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the growth rate of employees by job title over time.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing employment data
        
    Returns:
        pd.DataFrame: DataFrame containing job titles and their growth rates
    """
    pivot_table = df.pivot_table(
        values='tot_emp',
        index='occ_title',
        columns='year',
        aggfunc='sum',
        fill_value=0
    )
    
    first_year, last_year = pivot_table.columns.min(), pivot_table.columns.max()
    growth_rates = ((pivot_table[last_year] - pivot_table[first_year]) / pivot_table[first_year] * 100)
    
    growth_df = pd.DataFrame({
        'Occupation Title': growth_rates.index,
        'Growth Rate (%)': growth_rates.values,
        'First Year Count': pivot_table[first_year],
        'Last Year Count': pivot_table[last_year],
        'First Year': first_year,
        'Last Year': last_year
    })
    
    return (growth_df
            .dropna()
            .query('`First Year Count` > 0')
            .sort_values('Growth Rate (%)', ascending=False))

def calculate_salary_growth_rates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate salary growth rates for each occupation over time.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing salary data
        
    Returns:
        pd.DataFrame: DataFrame containing salary growth statistics by occupation
    """
    salary_trends = data[['occ_title', 'a_mean', 'year']].copy()
    salary_trends = salary_trends.sort_values(['occ_title', 'year'])
    salary_trends['wage_growth'] = salary_trends.groupby('occ_title')['a_mean'].pct_change(fill_method=None) * 100
    
    growth_stats = (salary_trends
                   .groupby('occ_title')
                   .agg({
                       'wage_growth': 'mean',
                       'a_mean': ['first', 'last'],
                       'year': ['first', 'last']
                   })
                   .round(2))
    
    growth_stats.columns = ['Growth Rate (%)', 'First Year Amount', 'Last Year Amount', 
                          'First Year', 'Last Year']
    
    growth_stats = growth_stats.reset_index()
    growth_stats = growth_stats.rename(columns={'occ_title': 'Occupation Title'})
    
    return growth_stats.sort_values('Growth Rate (%)', ascending=False)

def plot_top_growing_jobs(growth_analysis: pd.DataFrame, top_n: int = 10) -> Tuple[Figure, Axes]:
    """
    Create a bar plot of the top N fastest growing job titles.
    
    Args:
        growth_analysis (pd.DataFrame): DataFrame containing growth analysis results
        top_n (int): Number of top jobs to display
        
    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    top_jobs = growth_analysis.head(top_n)
    bars = ax.bar(range(top_n), top_jobs['Growth Rate (%)'])
    
    ax.set_title('Top 10 Fastest Growing Job Titles')
    ax.set_xlabel('Job Title')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_jobs['Occupation Title'], rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig, ax

def save_analysis_results(growth_analysis: pd.DataFrame, 
                         salary_growth_stats: pd.DataFrame,
                         output_dir: Optional[Path] = None) -> None:
    """
    Save analysis results to CSV files.
    
    Args:
        growth_analysis (pd.DataFrame): Employee growth analysis results
        salary_growth_stats (pd.DataFrame): Salary growth analysis results
        output_dir (Optional[Path]): Directory to save results. Defaults to Computed Data directory
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "Computed Data"
    output_dir.mkdir(exist_ok=True)
    
    # Save with consistent formatting
    growth_analysis.to_csv(output_dir / "employee_growth_rates.csv", index=False, float_format='%.2f')
    salary_growth_stats.to_csv(output_dir / "salary_growth_rates.csv", index=False, float_format='%.2f')
    print(f"\nAnalysis results saved to: {output_dir}")

def display_dataset_info(data: pd.DataFrame) -> None:
    """
    Display comprehensive information about the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame to analyze
    """
    print("\nDataset Information:")
    print("=" * 50)
    print(f"Number of rows: {len(data):,}")
    print(f"Number of columns: {len(data.columns)}")
    print("\nColumn names:", data.columns.tolist())
    
    print("\nData Types and Non-null Counts:")
    print("=" * 50)
    print(data.info())
    
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("\nMissing Values:")
        print("=" * 50)
        print(missing_values[missing_values > 0])

def main() -> None:
    """Main function to run all analyses."""
    try:
        print("\nLoading data...")
        data = load_standardized_data()
        display_dataset_info(data)
        
        print("\nAnalyzing employment growth rates...")
        print("=" * 50)
        growth_analysis = analyze_employee_growth(data)
        print(f"Found growth rates for {len(growth_analysis)} occupations")
        print(f"Time period: {growth_analysis['First Year'].iloc[0]} - {growth_analysis['Last Year'].iloc[0]}")
        
        print("\nAnalyzing salary growth rates...")
        print("=" * 50)
        salary_growth_stats = calculate_salary_growth_rates(data)
        print(f"Found growth rates for {len(salary_growth_stats)} occupations")
        print(f"Time period: {salary_growth_stats['First Year'].iloc[0]} - {salary_growth_stats['Last Year'].iloc[0]}")
        
        plot_top_growing_jobs(growth_analysis)
        save_analysis_results(growth_analysis, salary_growth_stats)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main()