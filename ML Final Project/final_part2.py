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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from final_part1 import (
    load_standardized_data,
    analyze_employee_growth,
    analyze_salary_growth
)

# MERGE EMPLOYEE AND SALARY GROWTH DATA
def merge_growth_data(emp_growth: pd.DataFrame, sal_growth: pd.DataFrame) -> pd.DataFrame:
    """Merges employee growth and salary growth dataframes."""
    required_emp_col = 'Mean Annual Growth (%)'
    required_sal_col = 'Growth Rate (%)'
    if required_emp_col not in emp_growth.columns:
        raise KeyError(f"Required column '{required_emp_col}' not found in employee growth DataFrame. Columns: {emp_growth.columns.tolist()}")
    if required_sal_col not in sal_growth.columns:
         raise KeyError(f"Required column '{required_sal_col}' not found in salary growth DataFrame. Columns: {sal_growth.columns.tolist()}")
    merged_data = pd.merge(
        emp_growth[['Occupation Title', required_emp_col]],
        sal_growth[['Occupation Title', required_sal_col]],
        on='Occupation Title',
    )
    merged_data = merged_data.rename(columns={
        required_emp_col: 'Employee Growth Rate (%)',
        required_sal_col: 'Salary Growth Rate (%)'
    })
    merged_data['Employee Growth Rate (%)'] = pd.to_numeric(merged_data['Employee Growth Rate (%)'], errors='coerce')
    merged_data['Salary Growth Rate (%)'] = pd.to_numeric(merged_data['Salary Growth Rate (%)'], errors='coerce')
    merged_data = merged_data.replace([np.inf, -np.inf], np.nan).dropna()
    return merged_data



# PLOT GROWTH CORRELATION FOR SPECIFIC GROUP
def plot_growth_correlation_for_group(data: pd.DataFrame,
                             group_name: str,
                             emp_growth_func,
                             sal_growth_func) -> tuple[Figure, Axes] | None:
    """Generates a scatter plot with line of best fit for a specific occupation group's growth rates."""
    print(f"\nGenerating growth correlation plot for Occupation Group: '{group_name}'...")

    # Filter data for the specific group
    if 'o_group' not in data.columns:
        print(f"ERROR: Column 'o_group' not found in the DataFrame.")
        return None

    data_filtered = data.dropna(subset=['o_group'])
    group_df = data_filtered[data_filtered['o_group'].str.lower() == group_name.lower()].copy()

    if group_df.empty:
        print(f"WARN: No data found for occupation group '{group_name}'. Skipping plot.")
        return None

    # Calculate growth rates for this group
    try:
        group_df['tot_emp'] = pd.to_numeric(group_df['tot_emp'], errors='coerce')
        if pd.api.types.is_object_dtype(group_df['a_mean']):
             group_df['a_mean'] = pd.to_numeric(group_df['a_mean'].astype(str).str.replace(',', ''), errors='coerce')
        else:
             group_df['a_mean'] = pd.to_numeric(group_df['a_mean'], errors='coerce')
        group_df = group_df.dropna(subset=['tot_emp', 'a_mean'])
    except Exception as e:
        print(f"WARN: Error preparing data for group '{group_name}': {e}. Skipping plot.")
        return None

    if group_df.empty or len(group_df['year'].unique()) < 2:
        print(f"WARN: Insufficient data or years for growth analysis within group '{group_name}'. Skipping plot.")
        return None

    # Calculate growth rates for this group
    try:
        emp_growth_group = emp_growth_func(group_df)
        sal_growth_group = sal_growth_func(group_df)
    except Exception as e:
        print(f"WARN: Error calculating growth rates for group '{group_name}': {e}. Skipping plot.")
        return None

    # Merge growth data for the group
    merged_group_data = merge_growth_data(emp_growth_group, sal_growth_group)

    emp_col = 'Employee Growth Rate (%)'
    sal_col = 'Salary Growth Rate (%)'

    if merged_group_data.empty or len(merged_group_data) < 2:
        print(f"WARN: Not enough valid data points after cleaning for group '{group_name}' to plot correlation. Skipping plot.")
        return None

    # Calculate correlation specific to this function scope
    try:
         correlation, p_value = pearsonr(merged_group_data[emp_col], merged_group_data[sal_col])
    except ValueError:
         correlation, p_value = np.nan, np.nan
         print(f"WARN: Could not calculate Pearson correlation for group '{group_name}'.")

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(merged_group_data[emp_col], merged_group_data[sal_col], alpha=0.6, label='Occupations')

    # Calculate and plot line of best fit
    m, b = np.polyfit(merged_group_data[emp_col], merged_group_data[sal_col], 1)
    x_line = np.array([merged_group_data[emp_col].min(), merged_group_data[emp_col].max()])
    y_line = m * x_line + b
    ax.plot(x_line, y_line, color='red', linestyle='--', label=f'Fit: y={m:.2f}x+{b:.2f}')

    # Add labels and title
    ax.set_title(f'Employee Growth vs Salary Growth ({group_name.capitalize()} Group, N={len(merged_group_data)})')
    ax.set_xlabel(emp_col)
    ax.set_ylabel(sal_col)
    ax.grid(True)
    ax.legend()

    # Add correlation text
    if not np.isnan(correlation):
        textstr = f'Pearson R: {correlation:.2f}\nP-value: {p_value:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig, ax


# ANALYZE ATTRIBUTE CORRELATIONS (LATEST YEAR)
def analyze_attribute_correlations(df: pd.DataFrame, year: int = 2023):
    """Calculates and visualizes correlations between numeric attributes for a specific year."""
    print(f"\nAnalyzing Attribute Correlations for the year {year}...")
    print("=" * 50)

    # Filter data for the specified year
    df_year = df[df['year'] == year].copy()

    if df_year.empty:
        print(f"WARN: No data found for the year {year}. Skipping attribute correlation analysis.")
        return

    # Select relevant numeric columns for correlation analysis
    numeric_cols = [
        'tot_emp', 'jobs_1000', 'loc_quotient', 'pct_total',
        'h_mean', 'h_median', 'h_pct10', 'h_pct25', 'h_pct75', 'h_pct90',
        'a_mean', 'a_median', 'a_pct10', 'a_pct25', 'a_pct75', 'a_pct90'
    ]
    
    valid_numeric_cols = []
    for col in numeric_cols:
        if col in df_year.columns:
            if pd.api.types.is_object_dtype(df_year[col]):
                 df_year[col] = pd.to_numeric(df_year[col].astype(str).str.replace(',', '').str.replace('*', '').str.replace('#', ''), errors='coerce')
            else:
                 df_year[col] = pd.to_numeric(df_year[col], errors='coerce')
            if not df_year[col].isnull().all():
                valid_numeric_cols.append(col)
        else:
            print(f"WARN: Column '{col}' not found in data for year {year}.")
            
    if not valid_numeric_cols:
        print(f"WARN: No valid numeric columns found for correlation analysis in year {year}.")
        return
        
    df_corr = df_year[valid_numeric_cols].dropna()

    if len(df_corr) < 2:
        print(f"WARN: Not enough valid data points (after dropping NaNs) for correlation analysis in year {year}.")
        return

    correlation_matrix = df_corr.corr()

    print("\nCorrelation Matrix (Selected Numeric Attributes):")
    print(correlation_matrix)

    # Visualize the correlation matrix using matplotlib
    try:
        plt.figure(figsize=(14, 12))
        plt.matshow(correlation_matrix, cmap='coolwarm', fignum=plt.gcf().number)
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.gca().xaxis.tick_bottom()
        plt.colorbar(label='Correlation')
        
        # Add annotations
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", 
                         ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black', 
                         fontsize=8)
        
        plt.title(f'Correlation Matrix of Numeric Attributes ({year})', pad=80)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    except Exception as e:
        print(f"WARN: Could not generate matplotlib correlation matrix plot. Error: {e}")

    # Plot scatter plots for selected pairs
    pairs_to_plot = [
        ('tot_emp', 'a_mean'), 
        ('a_pct10', 'a_pct90'),
        ('loc_quotient', 'a_mean')
    ]

    print("\nGenerating Scatter Plots for Selected Attribute Pairs...")
    for x_col, y_col in pairs_to_plot:
        if x_col in df_corr.columns and y_col in df_corr.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df_corr[x_col], df_corr[y_col], alpha=0.5)
                
                m, b = np.polyfit(df_corr[x_col], df_corr[y_col], 1)
                x_line = np.array([df_corr[x_col].min(), df_corr[x_col].max()])
                y_line = m * x_line + b
                ax.plot(x_line, y_line, color='red', linestyle='--', label=f'Fit: y={m:.2f}x+{b:.2f}')
                
                correlation_val = correlation_matrix.loc[x_col, y_col]
                ax.set_title(f'{x_col} vs {y_col} ({year}) (R={correlation_val:.2f})')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
            except Exception as e:
                 print(f"WARN: Could not generate scatter plot for '{x_col}' vs '{y_col}'. Error: {e}")
        else:
            print(f"WARN: Skipping plot for '{x_col}' vs '{y_col}' - one or both columns missing/invalid.")



# CREATE DATA PREPARATION PIPELINE
def create_preprocessing_pipeline(numeric_features: list, categorical_features: list):
    """Creates a scikit-learn pipeline for preprocessing numeric and categorical features."""
    print("\nDefining Preprocessing Pipeline Structure...")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    print("Numeric Transformer Steps: Imputer(median) -> StandardScaler")

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    print("Categorical Transformer Steps: Imputer(most_frequent) -> OneHotEncoder")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    print("ColumnTransformer applies these to specified numeric/categorical feature lists.")
    
    return preprocessor


# MAIN EXECUTION BLOCK
def main():
    try:
        print("\nLoading data...")
        data = load_standardized_data()

        # Analyze attribute correlations for latest year
        analyze_attribute_correlations(data, year=2023)

        # Define and display preprocessing pipeline structure
        # Define feature lists (Adjust based on your modeling goals)
        numeric_features = ['tot_emp', 'loc_quotient', 'a_mean', 'h_mean', 'a_median'] 
        categorical_features = ['o_group', 'area_type']
        
        # Filter to features actually present in the data
        existing_numeric = [col for col in numeric_features if col in data.columns]
        existing_categorical = [col for col in categorical_features if col in data.columns]

        if existing_numeric or existing_categorical:
            preprocessing_pipeline = create_preprocessing_pipeline(
                numeric_features=existing_numeric, 
                categorical_features=existing_categorical
            )
            print("\nPreprocessing pipeline object created:")
            print(preprocessing_pipeline)
        else:
            print("\nWARN: No suitable features found to define preprocessing pipeline.")

        # Plot correlation for specific groups (Growth Rates)
        plot_growth_correlation_for_group(data, 'major', analyze_employee_growth, analyze_salary_growth)
        plot_growth_correlation_for_group(data, 'minor', analyze_employee_growth, analyze_salary_growth)

        print("\nDisplaying all generated plots...")
        plt.show() # Show all plots at the end

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main()
