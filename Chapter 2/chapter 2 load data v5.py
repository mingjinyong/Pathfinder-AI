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





#Open up the file housing.csv and convert the data into a pandas DataFrame object
def load_housing_data():

    return pd.read_csv("Datasets\housing.csv")

#Create a new attribute income_cat
def do_the_cut(housing):
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.,1.5,3.,4.5,6., np.inf], labels=[1,2,3,4,5])
    #housing["income_cat"].hist()
    #plt.show() #You have to delete the popup window to continue the program

housing = load_housing_data()
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
print(housing_num.info())
num_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                          ('std_scaler', StandardScaler()),])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#We separate the attributes into the numeric ones and the text ones (just ocean-proximity)
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attribs), \
                                    ("cat", OneHotEncoder(), cat_attribs),])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing)



#Training and evaluating on the training set
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

'''
#Done. Now we have a working Linear Regression model. Let's try it out on a few instances from the training set
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
'''
'''
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
'''
'''
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#display the results
print("DT_Scores:", scores)
print("DT_Mean:", scores.mean())
print("DT_Standard deviation:", scores.std())

#Let's do cross validation on the linear regression model
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("LE_Scores:", lin_rmse_scores)
print("LE_Mean:", lin_rmse_scores.mean())
print("LE_Standard deviation:", lin_rmse_scores.std())
'''

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("Forest Scores:", forest_rmse_scores)
print("Forest Mean:", forest_rmse_scores.mean())
print("Forest Standard deviation:", forest_rmse_scores.std())
print("COMPLETED")

