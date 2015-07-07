
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf1 = GradientBoostingRegressor( n_estimators = 1950 , max_depth = 9 , max_features = 27)
        self.clf2 = AdaBoostRegressor(RandomForestRegressor(n_estimators=50, max_depth=50, max_features=25), n_estimators=20)
                
    def fit(self, X, y):
        self.clf1.fit(X, y)
        self.clf2.fit(X, y)
        
    def predict(self, X):
        return self.clf1.predict(X) * 0.9 + self.clf2.predict(X) * 0.1