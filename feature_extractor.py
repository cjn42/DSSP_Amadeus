import pandas as pd
import os

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        
        #uncomment the line below in the submission
        path = os.path.dirname(__file__)
        X_encoded = X_df
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
                
        data_magic = pd.read_csv(os.path.join(path, "data_magic_2.csv"))
        #data_magic = pd.read_csv("data_magic_2.csv")
        X_magic = data_magic[['DateOfDeparture','Departure','Arrival','Mean_Temp']]
        X_encoded = X_encoded.merge(X_magic, how='left', left_on=['DateOfDeparture','Departure','Arrival'], right_on=['DateOfDeparture','Departure','Arrival'], sort=False)
              
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
    
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        X_encoded = X_encoded.drop('weekday', axis=1)
        X_encoded = X_encoded.drop('week', axis=1)
        X_encoded = X_encoded.drop('year', axis=1)
        X_encoded = X_encoded.drop('std_wtd', axis=1)
        X_encoded = X_encoded.drop('WeeksToDeparture', axis=1)        
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)     
        X_array = X_encoded.values
        return X_array