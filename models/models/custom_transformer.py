from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd



class MessageExtractor(BaseEstimator, TransformerMixin):
    '''
        Input: X
        return: X['message']

        MessageExtractor is a transformer , can be used in pipeline to extract the message feature from the input dataframe.
    '''    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X["message"]#.values.reshape(-1,1)


class GenreExtractor(BaseEstimator, TransformerMixin):
    '''
        Input: X
        return: X['genre']

        GenreExtractor is a transformer , can be used in pipeline to extract the genre feature from the input dataframe.
    '''    

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X["genre"].values.reshape(-1,1)


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
        Input: X
        return: pandas series of length of text

        TextLengthExtractor is a transformer , can be used in pipeline to extract the length of the text from a given input.
        Input can be an array of text or pandas Series.
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(lambda x: len(x)).values.reshape(-1,1)


class WordCountExtractor(BaseEstimator, TransformerMixin):
    '''
        Input: X
        return: pandas series of word count

        WordCountExtractor is a transformer , can be used in pipeline to extract the number of words of the text from a given input.
        Input can be an array of text or pandas Series.
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(lambda x: len(x.split())).values.reshape(-1,1)

