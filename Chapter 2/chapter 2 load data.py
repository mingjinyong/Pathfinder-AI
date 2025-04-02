import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy
import sklearn

def load_housing_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

print(housing.head())

housing.info()

print(housing["ocean_proximity"].value_counts())

print(housing.describe())

housing.hist(bins=50, figsize=(20,15))

plt.show()

plt.figure(figsize=(10,6))
plt.hist(housing["median_house_value"], bins=50)
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.title("Distribution of Median House Values")
plt.show()
