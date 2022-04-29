from sklearn import preprocessing
import numpy as np

class PreProcessor():

    def __init__(self, X):
        self.X = X

    def rescale(self):
        scaler = preprocessing.StandardScaler().fit(self.X)
        X_scaled = scaler.transform(self.X)
        return X_scaled
