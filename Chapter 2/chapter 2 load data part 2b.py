import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from zlib import crc32  # Adding missing import for crc32 function

# Import scikit-learn components for data preprocessing and modeling
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Function to load the housing dataset from a CSV file
def load_housing_data():
    """
    Loads the housing dataset from a CSV file located in the same directory as this script.
    
    Returns:
        pandas.DataFrame: The housing dataset as a DataFrame
    """
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to housing.csv
    csv_path = os.path.join(current_dir, "houses.csv")
    # Read and return the CSV file as a pandas DataFrame
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    """
    Splits the dataset into training and test sets using random sampling.
    Note: This method picks indices at random each time you run it, which can lead to data leakage.
    
    Args:
        data (pandas.DataFrame): The dataset to split
        test_ratio (float): The proportion of the dataset to include in the test split (0-1)
    
    Returns:
        tuple: (train_set, test_set) as pandas DataFrames
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    """
    Determines if a specific instance should be in the test set based on its identifier.
    Uses CRC32 hash to ensure consistent assignment across multiple runs.
    
    Args:
        identifier: The unique identifier for the instance
        test_ratio (float): The proportion of the dataset to include in the test split (0-1)
    
    Returns:
        bool: True if the instance should be in the test set, False otherwise
    """
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    """
    Splits the dataset into training and test sets using a consistent approach based on instance IDs.
    This ensures the same instances end up in the same sets across multiple runs.
    
    Args:
        data (pandas.DataFrame): The dataset to split
        test_ratio (float): The proportion of the dataset to include in the test split (0-1)
        id_column (str): The name of the column containing unique identifiers
    
    Returns:
        tuple: (train_set, test_set) as pandas DataFrames
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def remove_income_cat(strat_train_set, strat_test_set):
    """
    Removes the 'income_cat' column from both training and test sets.
    This column is only used for stratified sampling and should be removed before model training.
    
    Args:
        strat_train_set (pandas.DataFrame): The stratified training set
        strat_test_set (pandas.DataFrame): The stratified test set
    """
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

def do_the_cut(housing):
    """
    Creates a new categorical attribute 'income_cat' based on median income.
    This is used for stratified sampling to ensure the test set is representative.
    
    Args:
        housing (pandas.DataFrame): The housing dataset
    """
    # Create income categories by binning median_income into 5 categories
    housing['income_cat'] = pd.cut(housing["median_income"], 
                                  bins=[0., 1.5, 3., 4.5, 6., np.inf], 
                                  labels=[1, 2, 3, 4, 5])
    # Display histogram of income categories
    housing["income_cat"].hist()
    plt.show()  # You have to close the popup window to continue the program

# Load the housing dataset
housing = load_housing_data()

# APPROACH 1: Simple random sampling for train/test split
# This approach doesn't guarantee the same split across multiple runs
train_set, test_set = split_train_test(housing, 0.2)

# APPROACH 2: Consistent train/test split using instance IDs
# This ensures the same instances end up in the same sets across multiple runs
housing_with_id = housing.reset_index()  # Adds an index column to use as ID
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
print(len(train_set))
print(len(test_set))

# Create income categories for stratified sampling
do_the_cut(housing)

# APPROACH 3 (FINAL): Stratified sampling for train/test split
# This ensures the distribution of income categories is the same in both sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

# Removes the income category we added to the original dataset
remove_income_cat(strat_train_set, strat_test_set)

# Now we just want to use the training set. Test set will be needed once we have ML algo trained
housing = strat_train_set.copy()
print(len(housing))

housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show() # This is required in PyCharm - not shown in book

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()
plt.show() # This is required in PyCharm - not shown in book

corr_matrix = housing.corr(numeric_only = True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

# Feature engineering - creating new attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Remove median_house_value as an attribute - since this is our label attribute
housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()

print(housing.info())
print(housing_labels)
print(strat_test_set.head(5))
print(strat_train_set.head(5))

# 3 options for dealing with missing total_bedrooms
# Option 1: Drop rows with missing values
housing_option1 = housing.copy()
housing_option1 = housing_option1.dropna(subset=["total_bedrooms"])
housing_option1.info()

# Option 2: Drop the entire column
housing_option2 = housing.copy()
housing_option2 = housing_option2.drop("total_bedrooms", axis=1)
housing_option2.info()

# Option 3: Fill missing values with the median
housing_option3 = housing.copy()
median = housing_option3["total_bedrooms"].median()
housing_option3["total_bedrooms"].fillna(median, inplace=True)
housing_option3.info()

# Data preprocessing for numerical features using scikit-learn's SimpleImputer
housing_num = housing.drop("ocean_proximity", axis=1)
imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
x = imputer.transform(housing_num)
housing_tr = pd.DataFrame(x, columns=housing_num.columns)
print(housing_tr.info())

# Handling categorical features using OrdinalEncoder
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(ordinal_encoder.categories_)  # Text categories and their ordinal value

# Creating a preprocessing pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)

# Creating a full preprocessing pipeline for both numerical and categorical features
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing)