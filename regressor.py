from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
 
class Regressor(BaseEstimator):
    def __init__(self):
        self.clf1 = GradientBoostingRegressor( n_estimators = 1950 , max_depth = 9 , max_features = 27)
        self.clf2 = xgb.XGBRegressor(max_depth=17, n_estimators=1000, learning_rate=0.05)
        
    def fit(self, X, y):
        self.clf1.fit(X, y)
        self.clf2.fit(X, y)     
 
    def predict(self, X):
        return self.clf1.predict(X) * 0.6 + self.clf2.predict(X) * 0.4