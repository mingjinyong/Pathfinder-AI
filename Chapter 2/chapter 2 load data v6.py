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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import customtransformer as cf


#imports for v6
from sklearn.model_selection import GridSearchCV



#Open up the file housing.csv and convert the data into a pandas DataFrame object
def load_housing_data():
    return pd.read_csv("Datasets\housing.csv")

#Create a new attribute income_cat
def do_the_cut(housing):
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.,1.5,3.,4.5,6., np.inf], labels=[1,2,3,4,5])
    #housing["income_cat"].hist()
    #plt.show() #You have to delete the popup window to continue the program

housing = load_housing_data()
#housing = pd.read_csv("mydataset.csv")
#housing.info()

do_the_cut(housing)

#Final approach for creating reliable train/test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#Now we just want to use the training set. Test set will be needed once we have ML algo trained
housing = strat_train_set.copy()

#We need to remove the label attribute - our answer - before cleaning up the data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#We need to remove the text attribute so that we can use num_pipeline on the numberic data and OneHotEncoder on ocean-proximity
housing_num = housing.drop("ocean_proximity", axis=1)

num_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                          ('attribs_adder', cf.CombinedAttributeAdder()),
                          ('std_scaler', StandardScaler()),])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#We separate the attributes into the numeric ones and the text ones (just ocean-proximity)
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attribs), \
                                    ("cat", OneHotEncoder(), cat_attribs),])

housing_prepared = full_pipeline.fit_transform(housing)
#print(housing_prepared)
#print(housing)

#-----------------------------------
#Fine Tune your model
param_grid = [
#    {'n_estimators': [5,15,30], 'max_features': [2,4,8,15]},
    {'bootstrap': [False], 'n_estimators': [20, 40, 80], 'max_features': [2,8,15,20]},]


forest_reg = RandomForestRegressor()

grid_search = GridSearchCV( forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit( housing_prepared, housing_labels)

print(grid_search.best_params_)


print(grid_search.best_estimator_)
'''
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score), params)
'''

#feature_importances = grid_search.best_estimator_.feature_importances_
#print(feature_importances)

'''
extra_attribs = ["rooms_per_hhold","pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))
'''


final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

