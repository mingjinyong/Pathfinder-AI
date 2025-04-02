import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32

def load_housing_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio): #picks indices at random each time you run it
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def do_the_cut(housing):
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.,1.5,3.,4.5,6., np.inf], labels=[1,2,3,4,5])
    housing["income_cat"].hist()
    plt.show()

housing = load_housing_data()
train_set, test_set = split_train_test(housing, 0.2)

housing_with_id = housing.reset_index()  #adds an index column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2,'index')

print(len(train_set))
print(len(test_set))
print(test_set.head())

housing.hist( bins=50, figsize=(20,15)) #shows the number of instances (vertical axis) that have a given value range
plt.show()#Plots a histogram for each numerical attribute

do_the_cut(housing)
