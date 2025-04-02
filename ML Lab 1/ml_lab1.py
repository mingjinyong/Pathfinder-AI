'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
ML Lab 1
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pandas.plotting import scatter_matrix

# Load the remote work productivity dataset
def load_work_data():
    """Loads the remote work productivity dataset from CSV"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "remote_work_productivity.csv")
    return pd.read_csv(csv_path)

# Load and explore data
work_data = load_work_data()
print("Dataset Information:")
print(f"Number of rows: {len(work_data)}")
print(f"Number of columns: {len(work_data.columns)}")
print("\nColumn names:", work_data.columns.tolist())
print("\nMissing values:")
print(work_data.isnull().sum())

# Split data into training and test sets
X = work_data.drop('Productivity_Score', axis=1)
y = work_data['Productivity_Score']
train_set, test_set, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {len(train_set)}")
print(f"Testing set size: {len(test_set)}")

# DATA PREPARATION

# Create efficiency ratio feature
train_set['Efficiency_Ratio'] = y_train / train_set['Hours_Worked_Per_Week']
test_set['Efficiency_Ratio'] = y_test / test_set['Hours_Worked_Per_Week']

# Analyze correlations
train_set_with_target = train_set.copy()
train_set_with_target['Productivity_Score'] = y_train
corr_matrix = train_set_with_target.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix["Productivity_Score"].sort_values(ascending=False))

# Visualize key relationships
attributes = ["Productivity_Score", "Hours_Worked_Per_Week", "Well_Being_Score", "Efficiency_Ratio"]
scatter_matrix(train_set_with_target[attributes], figsize=(12, 8))
plt.show()

# Hours worked vs productivity
plt.figure(figsize=(10, 6))
plt.scatter(train_set_with_target["Hours_Worked_Per_Week"], train_set_with_target["Productivity_Score"], alpha=0.5)
plt.xlabel("Hours Worked Per Week")
plt.ylabel("Productivity Score")
plt.title("Hours Worked vs Productivity")
plt.show()

# Compare Remote vs In-Office workers
plt.figure(figsize=(15, 6))
# Remote workers
remote_data = train_set_with_target[train_set_with_target['Employment_Type'] == 'Remote']
plt.subplot(1, 2, 1)
plt.scatter(remote_data["Hours_Worked_Per_Week"], remote_data["Productivity_Score"], 
            alpha=0.5, color='blue', label='Remote Workers')

# Add line of best fit for remote workers
remote_z = np.polyfit(remote_data["Hours_Worked_Per_Week"], remote_data["Productivity_Score"], 1)
remote_p = np.poly1d(remote_z)
plt.plot(np.sort(remote_data["Hours_Worked_Per_Week"]), 
         remote_p(np.sort(remote_data["Hours_Worked_Per_Week"])), 
         "b--", linewidth=2, label='Best Fit')
plt.xlabel("Hours Worked Per Week")
plt.ylabel("Productivity Score")
plt.title("Remote Workers: Hours vs Productivity")
plt.legend()

# In-office workers
inoffice_data = train_set_with_target[train_set_with_target['Employment_Type'] == 'In-Office']
plt.subplot(1, 2, 2)
plt.scatter(inoffice_data["Hours_Worked_Per_Week"], inoffice_data["Productivity_Score"], 
            alpha=0.5, color='green', label='In-Office Workers')

# Add line of best fit for in-office workers
inoffice_z = np.polyfit(inoffice_data["Hours_Worked_Per_Week"], inoffice_data["Productivity_Score"], 1)
inoffice_p = np.poly1d(inoffice_z)
plt.plot(np.sort(inoffice_data["Hours_Worked_Per_Week"]), 
         inoffice_p(np.sort(inoffice_data["Hours_Worked_Per_Week"])), 
         "g--", linewidth=2, label='Best Fit')
plt.xlabel("Hours Worked Per Week")
plt.ylabel("Productivity Score")
plt.title("In-Office Workers: Hours vs Productivity")
plt.legend()

plt.tight_layout()
plt.show()

# Prepare data for ML algorithms
train_set = train_set_with_target.drop("Productivity_Score", axis=1)
numeric_features = ['Employee_ID', 'Hours_Worked_Per_Week', 'Well_Being_Score', 'Efficiency_Ratio']
categorical_features = ['Employment_Type']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

full_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply preprocessing
X_train_prepared = full_pipeline.fit_transform(train_set)
X_test_prepared = full_pipeline.transform(test_set)
print("\nPreprocessed training set shape:", X_train_prepared.shape)

# TRAIN AND EVALUATE MODELS
# Linear Regression
print("\n--- Linear Regression Model ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

# Evaluate on training set
train_predictions = lin_reg.predict(X_train_prepared)
lin_rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
print(f"Training RMSE: {lin_rmse_train:.2f}")

# Cross-validation
lin_scores = cross_val_score(lin_reg, X_train_prepared, y_train, 
                            scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(f"Cross-validation RMSE - Mean: {lin_rmse_scores.mean():.2f}, Std: {lin_rmse_scores.std():.2f}")

# Random Forest Regression
print("\n--- Random Forest Regression Model ---")
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train_prepared, y_train)

# Evaluate on training set
train_predictions = forest_reg.predict(X_train_prepared)
forest_rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
print(f"Training RMSE: {forest_rmse_train:.2f}")

# Cross-validation
forest_scores = cross_val_score(forest_reg, X_train_prepared, y_train, 
                               scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(f"Cross-validation RMSE - Mean: {forest_rmse_scores.mean():.2f}, Std: {forest_rmse_scores.std():.2f}")

# EVALUATE FINAL MODELS ON TEST SET
# Linear Regression
print("\n--- Final Linear Regression Evaluation ---")
final_lin_predictions = lin_reg.predict(X_test_prepared)
final_lin_rmse = np.sqrt(mean_squared_error(y_test, final_lin_predictions))
final_lin_mae = mean_absolute_error(y_test, final_lin_predictions)
final_lin_r2 = r2_score(y_test, final_lin_predictions)

print(f"Test RMSE: {final_lin_rmse:.2f}")
print(f"Test MAE: {final_lin_mae:.2f}")
print(f"Test R²: {final_lin_r2:.2f}")

# Random Forest
print("\n--- Final Random Forest Evaluation ---")
final_forest_predictions = forest_reg.predict(X_test_prepared)
final_forest_rmse = np.sqrt(mean_squared_error(y_test, final_forest_predictions))
final_forest_mae = mean_absolute_error(y_test, final_forest_predictions)
final_forest_r2 = r2_score(y_test, final_forest_predictions)

print(f"Test RMSE: {final_forest_rmse:.2f}")
print(f"Test MAE: {final_forest_mae:.2f}")
print(f"Test R²: {final_forest_r2:.2f}")

# Compare models
print("\n--- Model Comparison ---")
print(f"Linear Regression Test RMSE: {final_lin_rmse:.2f}, R²: {final_lin_r2:.2f}")
print(f"Random Forest Test RMSE: {final_forest_rmse:.2f}, R²: {final_forest_r2:.2f}")

if final_forest_r2 > final_lin_r2:
    print("Random Forest Regression performed better on the test set.")
else:
    print("Linear Regression performed better on the test set.")

# Feature importance for Random Forest
if hasattr(forest_reg, 'feature_importances_'):
    cat_features = full_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out([categorical_features[0]])
    feature_names = numeric_features + list(cat_features)
    
    importances = forest_reg.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature ranking:")
    for f in range(min(len(feature_names), len(indices))):
        if f < len(indices):
            print(f"{f+1}. Feature '{feature_names[indices[f]]}' ({importances[indices[f]]:.4f})")

# Visualize predictions vs actual values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, final_lin_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Linear Regression: Predicted vs Actual')
plt.xlabel('Actual Productivity Score')
plt.ylabel('Predicted Productivity Score')

plt.subplot(1, 2, 2)
plt.scatter(y_test, final_forest_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest: Predicted vs Actual')
plt.xlabel('Actual Productivity Score')
plt.ylabel('Predicted Productivity Score')

plt.tight_layout()
plt.show()

# Residual plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
residuals_lr = y_test - final_lin_predictions
plt.scatter(final_lin_predictions, residuals_lr, alpha=0.5)
plt.hlines(y=0, xmin=final_lin_predictions.min(), xmax=final_lin_predictions.max(), colors='r', linestyles='--')
plt.title('Linear Regression: Residuals')
plt.xlabel('Predicted Productivity Score')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
residuals_rf = y_test - final_forest_predictions
plt.scatter(final_forest_predictions, residuals_rf, alpha=0.5)
plt.hlines(y=0, xmin=final_forest_predictions.min(), xmax=final_forest_predictions.max(), colors='r', linestyles='--')
plt.title('Random Forest: Residuals')
plt.xlabel('Predicted Productivity Score')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()
