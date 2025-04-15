"""
Ming Jin Yong & Johnny Sheng
ACAD 222, Spring 2025
mingjiny@usc.edu & johnnysh@usc.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import pearsonr

from final_part1 import (
    load_standardized_data,
    analyze_employee_growth,
    analyze_salary_growth
)


# MERGE EMPLOYEE AND SALARY GROWTH DATA
def merge_growth_data(emp_growth: pd.DataFrame, sal_growth: pd.DataFrame) -> pd.DataFrame:
    """Merges employee growth and salary growth dataframes."""
    # Check if necessary columns exist
    required_emp_col = 'Mean Annual Growth (%)' # Corrected column name
    required_sal_col = 'Growth Rate (%)'
    
    if required_emp_col not in emp_growth.columns:
        raise KeyError(f"Required column '{required_emp_col}' not found in employee growth DataFrame. Columns: {emp_growth.columns.tolist()}")
    if required_sal_col not in sal_growth.columns:
         raise KeyError(f"Required column '{required_sal_col}' not found in salary growth DataFrame. Columns: {sal_growth.columns.tolist()}")

    merged_data = pd.merge(
        emp_growth[['Occupation Title', required_emp_col]], # Use corrected column name
        sal_growth[['Occupation Title', required_sal_col]],
        on='Occupation Title',
        # Suffixes are not strictly needed if we rename immediately after
    )
    # Rename columns for clarity
    merged_data = merged_data.rename(columns={
        required_emp_col: 'Employee Growth Rate (%)', # Rename the corrected column
        required_sal_col: 'Salary Growth Rate (%)'
    })
    
    # Convert growth columns to numeric, coercing errors to NaN
    merged_data['Employee Growth Rate (%)'] = pd.to_numeric(merged_data['Employee Growth Rate (%)'], errors='coerce')
    merged_data['Salary Growth Rate (%)'] = pd.to_numeric(merged_data['Salary Growth Rate (%)'], errors='coerce')
    
    # Remove rows with infinite or NaN values (including those from coercion)
    merged_data = merged_data.replace([np.inf, -np.inf], np.nan).dropna()
    return merged_data


# CALCULATE CORRELATION BETWEEN EMPLOYEE AND SALARY GROWTH
def calculate_growth_correlation(merged_data: pd.DataFrame) -> tuple[float, float]:
    """Calculates the Pearson correlation between employee and salary growth rates."""
    
    # Ensure columns are numeric before calculating correlation
    emp_col = 'Employee Growth Rate (%)'
    sal_col = 'Salary Growth Rate (%)'
    if not pd.api.types.is_numeric_dtype(merged_data[emp_col]):
        raise TypeError(f"Column '{emp_col}' must be numeric for correlation calculation, but it has dtype {merged_data[emp_col].dtype}")
    if not pd.api.types.is_numeric_dtype(merged_data[sal_col]):
        raise TypeError(f"Column '{sal_col}' must be numeric for correlation calculation, but it has dtype {merged_data[sal_col].dtype}")
        
    if merged_data.empty or len(merged_data) < 2:
        print("Warning: Not enough valid data points after cleaning to calculate correlation.")
        return np.nan, np.nan
        
    correlation, p_value = pearsonr(
        merged_data[emp_col],
        merged_data[sal_col]
    )
    return correlation, p_value


# PLOT THE CORRELATION BETWEEN EMPLOYEE AND SALARY GROWTH
def plot_growth_correlation(merged_data: pd.DataFrame, correlation: float, p_value: float) -> tuple[Figure, Axes]:
    """Generates a scatter plot showing the correlation between employee and salary growth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(merged_data['Employee Growth Rate (%)'], merged_data['Salary Growth Rate (%)'], alpha=0.5)
    
    ax.set_title('Correlation between Employee Growth and Salary Growth')
    ax.set_xlabel('Employee Growth Rate (%)')
    ax.set_ylabel('Salary Growth Rate (%)')
    ax.grid(True)
    
    # Add correlation coefficient and p-value to the plot
    textstr = f'Pearson Correlation: {correlation:.2f}\nP-value: {p_value:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    return fig, ax


# MAIN EXECUTION BLOCK
def main():
    try:
        print("\nLoading data...")
        data = load_standardized_data()

        print("\nAnalyzing employment growth rates...")
        emp_growth = analyze_employee_growth(data)

        print("\nAnalyzing salary growth rates...")
        sal_growth = analyze_salary_growth(data)
        
        print("\nMerging growth data...")
        merged_growth_data = merge_growth_data(emp_growth, sal_growth)
        print(f"Merged data has {len(merged_growth_data)} occupations with both growth rates.")

        print("\nCalculating correlation...")
        correlation, p_value = calculate_growth_correlation(merged_growth_data)
        if not np.isnan(correlation):
             print(f"Pearson Correlation: {correlation:.2f}, P-value: {p_value:.3f}")
        else:
            print("Could not calculate correlation.")

        # Plot correlation
        if not merged_growth_data.empty and not np.isnan(correlation):
            print("\nPlotting correlation...")
            plot_growth_correlation(merged_growth_data, correlation, p_value)
            plt.show()
        else:
             print("Skipping correlation plot due to insufficient data.")

        print("\nAnalysis complete!")

    except Exception as e:
        print("\nError during analysis: {str(e)}")
        raise


main()

