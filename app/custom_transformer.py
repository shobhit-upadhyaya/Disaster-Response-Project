from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd



class MessageExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X["message"]#.values.reshape(-1,1)


class GenreExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X["genre"].values.reshape(-1,1)



class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(lambda x: len(x)).values.reshape(-1,1)


class WordCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(lambda x: len(x.split())).values.reshape(-1,1)

