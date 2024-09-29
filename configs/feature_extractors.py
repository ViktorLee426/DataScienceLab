import numpy as np
import pandas as pd
import os


class RocketFeatureExtractorOverlapping:
    def __init__(self):
        # Load pre-computed features from the file, to plug this in
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorOverlapping/X_train.csv")
        self.y_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorOverlapping/y_train.csv")
        self.X_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorOverlapping/X_test.csv")
        self.y_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorOverlapping/y_test.csv")

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    
class RocketFeatureExtractorNoOverlapping:
    def __init__(self):
        # Load pre-computed features from the file, to plug this in
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorNoOverlapping/X_train.csv")
        self.y_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorNoOverlapping/y_train.csv")
        self.X_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorNoOverlapping/X_test.csv")
        self.y_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/RocketExtractorNoOverlapping/y_test.csv")

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test


class VAEFeatureExtractorOverlapping:
    def __init__(self):
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractorOverlapping/X_train.csv")
        self.y_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractorOverlapping/y_train.csv")
        self.X_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractorOverlapping/X_test.csv")
        self.y_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractorOverlapping/y_test.csv")

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    
class VAEFeatureExtractor2:
    def __init__(self):
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractor2/X_train.csv")
        self.y_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractor2/y_train.csv")
        self.X_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractor2/X_test.csv")
        self.y_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/VAEExtractor2/y_test.csv")

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    


class LSTMFeatureExtractor:
    def __init__(self, n_estimators=100, n_jobs=-1):
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/LSTMExtractor/X_train.csv")
        self.y_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/LSTMExtractor/y_train.csv")
        self.X_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/LSTMExtractor/X_test.csv")
        self.y_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/LSTMExtractor/y_test.csv")

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    

class CNNFeatureExtractor:
    def __init__(self, n_estimators=100, n_jobs=-1):
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/CNNExtractor/X_train.csv")
        self.y_train = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/CNNExtractor/y_train.csv")
        self.X_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/CNNExtractor/X_test.csv")
        self.y_test = pd.read_csv("C:/Users/zheng/Desktop/DSLCodeBase/data/CNNExtractor/y_test.csv")

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

