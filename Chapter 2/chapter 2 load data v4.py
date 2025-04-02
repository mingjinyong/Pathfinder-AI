import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#The imports below are new for 4/7 lecture
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
#imnports for 4/9 lecture
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

def load_housing_data():
    return pd.read_csv("housing.csv")
'''
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

def remove_income_cat(strat_train_set, strat_test_set):
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
'''

#Create a new attribute income_cat
def do_the_cut(housing):
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.,1.5,3.,4.5,6., np.inf], labels=[1,2,3,4,5])
    #housing["income_cat"].hist()
    #plt.show() #You have to delete the popup window to continue the program

housing = load_housing_data()

#First attempt to create train/test sets
#train_set, test_set = split_train_test(housing, 0.2)

#second attempt to create train/test sets
##housing_with_id = housing.reset_index()  #adds an index column
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2,'index')
#print(len(train_set))
#print(len(test_set))

do_the_cut(housing)

#Final approach for creating reliable train/test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
#print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

#Removes the income category we added to the original dataset
#remove_income_cat(strat_train_set, strat_test_set)

#Now we just want to use the training set. Test set will be needed once we have ML algo trained
housing = strat_train_set.copy()
#rint(len(housing))

#housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)
#plt.show() #This is required in PyCharm - not shown in book

#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
#plt.legend()
#plt.show() #This is required in PyCharm - not shown in book

#corr_matrix = housing.corr(numeric_only=True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

#attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
#scatter_matrix(housing[attributes],figsize=(12,8))
#plt.show()

#housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.show()


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))


#housing = strat_train_set.drop("median_house_value", axis=1) #This is what we want to predict
#housing_labels = strat_train_set["median_house_value"].copy() #This creates a single DataFrame of our labels

#print(housing.info())
#print(housing_labels)

#3 options for dealing with missing total_bedrooms
#dropna returns the updated DataFrame object
#housing = housing.dropna(subset=["total_bedrooms"]) #option 1 from slides
#housing.info()

#housing = housing.drop("total_bedrooms", axis=1)
#housing.info()

#median=housing["total_bedrooms"].median()
#housing["total_bedrooms"].fillna(median, inplace=True)
#housing.info()

'''
housing_num = housing.drop("ocean_proximity", axis=1) #Must remove as it is not numeric
imputer = SimpleImputer(strategy="median") #Create a SimpleImputer object
imputer.fit(housing_num) #This does the work to compute the median of all attributes
print(imputer.statistics_) #The output from lines 124 and 125 should be the same
print(housing_num.median(numeric_only=True).values)
x = imputer.transform(housing_num) #This applies the median to any missing values
housing_tr = pd.DataFrame(x, columns=housing_num.columns) #convert our Pandas array back to a dataframe object
print(housing_tr.info())
'''

''' 
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(ordinal_encoder.categories_) #Text categories and their ordinal value
'''

''' 
num_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy="median")), \
               ('std_scaler', StandardScaler()),]) # we fix the missing values with amedian in total_bedrooms and change ocen_proximity to ordinals
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)
'''

'''
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attribs), \
                                    ("cat", OneHotEncoder(), cat_attribs),])
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing)
'''