from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

rooms_ix = 3
bedrooms_ix = 4
population_ix = 5
households_ix = 6

class CombinedAttributeAdder( BaseEstimator, TransformerMixin ):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, x, y=None):
        return self #nothing else to do

    def transform(self, x, y=None):
        rooms_per_household = x[:, rooms_ix] / x[:, households_ix]
        population_per_household = x[:, population_ix] / x[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:,bedrooms_ix] / x[:,rooms_ix]
            return np.c_[x,rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[x,rooms_per_household, population_per_household]

