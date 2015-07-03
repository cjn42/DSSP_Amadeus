#FE v2.0 : 
# suppression des variables de classes (replaced by get_dummies)
# suppression des mois et des N° de jour dans le mois (inutiles vs week et weekday)-> dim 100 vs 150
# ajout des variables pour les jours fériés significatifs (Xmas, NYD, Thg, Ind)

import pandas as pd


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
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
    
        data_weather = pd.read_csv("data_holidays.csv")
        X_weather = data_weather[['DateOfDeparture','Xmas','Xmas-1','NYD','NYD-1','Ind','Thg','Thg+1']]
        X_encoded = X_encoded.set_index(['DateOfDeparture'])
        X_weather = X_weather.set_index(['DateOfDeparture'])
        X_encoded = X_encoded.join(X_weather).reset_index()        
        
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