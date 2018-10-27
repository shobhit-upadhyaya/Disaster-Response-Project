# import libraries

import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, classification_report , accuracy_score, make_scorer, precision_score, recall_score
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from custom_transformer import MessageExtractor, GenreExtractor, TextLengthExtractor, WordCountExtractor
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")


def draw_evaluation_plots(old_results, new_results):
    '''
        Input: evaluation results of base and tunned model
        Return: None
    
        Desc:
            Creates a pointplot by comparing the results of two models.
            Generates plots of Accuracy, F1-Score, Precision and Recall and save as png images
    '''

    for col in old_results.columns:
        plt.figure(figsize=(15,5))
        plt.xticks(rotation=90)
        plt.title(col+" "+" plot")

        sns_plot = sns.pointplot(x = old_results.index, y = col, color='r', data=old_results)
        sns_plot = sns.pointplot(x = new_results.index, y = col, color='g', data=new_results)
        plt.tight_layout(True)
        
        sns_plot.figure.savefig("./img/"+ col +".png")


def load_data(database_filepath):
    '''
        Input: database_filepath
        return : X, y and category_names

        Desc:

            Creates sqlite engine.
            Reads the data of sql table and load it as dataframe.
            Load dataset from database by using read_sql_table()
            Generates X, y and category_names
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DiasterResponseData', engine)

    category_names = list(df.columns)[4:]

    X = df['message'].values
    y = df[category_names].values

    return X, y, category_names


def tokenize(text):
    '''
        Input: text
        Returns: clean tokens
        Desc:

            Generates a clean token of text (words) by first getting words from the text.
            Applies Lemmatization on the words.
            Normalize the text by lowering it and removes the extra spaces.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def multioutput_f1_score(y_test, y_pred):
    '''
        Input: y_test and y_pred
        Return: mean f1_score

        Desc:
            multioutput_f1_score function is a custom scoring function that is used by GridSearchCV for finding best results.
            For all categories it first finds the F1-Score and returns the mean of all the F1-Scores.
    '''

    f1_scores = []
    max_categories = len(y_test[0])
    #print(max_categories)
    #print(y_test.shape)
    for i in range(max_categories):
        res = f1_score(y_test[:,i], y_pred[:,i], average='weighted')
        f1_scores.append(res)
    #print(np.mean(f1_scores))
    return np.mean(f1_scores)


def build_model_tunned():
    '''
        Input: None
        Returns: CV (GridSearch cross validator)

        Desc:
            build_model_tunned function() creates a pipeline by using FeatureUnion.
            Features:
                BOW features by Text transformation
                TextLength Extractor
                WordCount Extractor

            Estimator:
                XGBClassifier

            GridSearchCV : for fine tunning pipeline hyper-parameters.

            For a scoring function , we are using custom mean F1-Score function that satisfies the criteria for 
            selecting best hyper-parameter and best-estimators.

            Returns cv after finding the best parameters and estimators.
    '''
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_length_pipeline', Pipeline([
                ('text_len', TextLengthExtractor()),
                
            ])),

            ('word_count_pipeline', Pipeline([
                ('word_count', WordCountExtractor()),                
            ])),

        ])),

        ('clf', MultiOutputClassifier( XGBClassifier(seed=1)) )
        
    ])  


    F1_SCORE_M = make_scorer(multioutput_f1_score, greater_is_better=True)

    parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 2),),
    'features__text_pipeline__vect__max_df': (0.5,),
    'features__text_pipeline__vect__max_features': [5000,],
    'features__text_pipeline__tfidf__use_idf': (False,),
    
    'clf__estimator__n_estimators': [200,],
  
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=0, scoring=F1_SCORE_M, n_jobs=-1)
    return cv


def build_model():
    '''
        Input: None
        Returns: pipeline object

        Desc:
            Base model function() creates a pipeline of text transformation with randomforest classifier as an estimator.
            Returns: pipeline 
    '''
    
    pipeline = Pipeline([
        (   
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))

        ),

        ('clf', MultiOutputClassifier( RandomForestClassifier(random_state=1)) )
        
    ])  


    #F1_SCORE_M = make_scorer(multioutput_f1_score, greater_is_better=True)

    #parameters = {}

    #cv = GridSearchCV(pipeline, param_grid=parameters, verbose=0, scoring=F1_SCORE_M, n_jobs=-1)




    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Input: model, X_test, y_test , category_names
        Return: dataframe with evaluation results

        Desc:
            evaluate_model function evaluates the performance of the trained model.
            Takes model, X_test , y_test and category_names as input and returns the evaluation dataframe 
            

            that contains the information about Accuracy, F1-Score, Precision and Recall

    '''
    y_pred = model.predict(X_test)
    f1_score_results = []
    precision_results = []
    recall_results = []
    accuracy_results = []

    results_df = pd.DataFrame(index=category_names, columns=['Accuracy','F1-Score','Precision', 'Recall'])

    for category_index , category_name in enumerate(category_names):

        accuracy_res = accuracy_score(Y_test[:, category_index], y_pred[:, category_index])
        f1_score_res = f1_score(Y_test[:, category_index], y_pred[:, category_index], average='weighted')
        precision_res = precision_score(Y_test[:, category_index], y_pred[:, category_index], average='weighted')
        recall_res = recall_score(Y_test[:, category_index], y_pred[:, category_index], average='weighted')

        accuracy_results.append(accuracy_res)
        precision_results.append(precision_res)
        recall_results.append(recall_res)
        f1_score_results.append(f1_score_res)

    results_df['Accuracy'] = accuracy_results
    results_df['F1-Score'] = f1_score_results
    results_df['Precision'] = precision_results
    results_df['Recall'] = recall_results    

    print("\n=========================================================================================\n") 
    print(results_df)
    print("\n=========================================================================================\n")
    
    print("Overall results: ")
    print(results_df.mean(axis=0))    
    return results_df


def save_model(model, model_filepath):
    """
        Input: model, model_filepath
        Return : None

        Desc:
            save_model() function saves the trained model to disk for future use.
    """
    with open(model_filepath, 'wb') as fObj:
        pickle.dump(model, fObj)


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        print('Building basic model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
     
        print('Evaluating model...')
        base_model_results = evaluate_model(model, X_test, Y_test, category_names)
        print(base_model_results.head())

        # Tunned model 
        print('Building tunned model...')
        model = build_model_tunned()
        
        print('Training model...')
        model.fit(X_train, Y_train)
     
        print('Evaluating model...')
        tunned_model_results = evaluate_model(model, X_test, Y_test, category_names)
        print(tunned_model_results.head())

        draw_evaluation_plots(base_model_results, tunned_model_results)

        print(model.best_params_)
        print(model.best_score_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()