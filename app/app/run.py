import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter, Layout
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.figure_factory as ff

import numpy as np


import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from custom_transformer import MessageExtractor, GenreExtractor, TextLengthExtractor, WordCountExtractor


app = Flask(__name__)


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
    for i in range(max_categories):
        res = f1_score(y_test[:,i], y_pred[:,i], average='weighted')
        f1_scores.append(res)
    return np.mean(f1_scores)


def bubble_line_plot_graph(genre_names,genre_counts):
    '''
        Input: genre_names, genre_counts
        Return : graph
        Desc:
            Creates a joint graph of Bar and Line plots to demonstrate the distribution Genre and Message
    '''
    colors = [0.36673455, 0.89371485, 0.11163017]  #np.random.rand(N)
    sz = genre_counts / 100    
    print(colors)

    bubble_line_graph = [{
    "data": [
        Scatter(x=genre_names, 
            y=genre_counts,
            mode = 'markers',
            name = 'BubblePlot:  of Genres Message Group',
            marker={'size': sz.values,
                        'color': colors,
                        'opacity': 0.6,
                        'colorscale': 'Viridis'
                       }
            ),

        Scatter(x=genre_names, 
            y=genre_counts,
            name = "LinePlot: of Genres Message Distribution",
            line = dict(
                color = ('rgb(22, 96, 167)'),
                width = 4,
                dash = 'dashdot')
            )
        
            
        ],
    'layout': {
                'title': 'BubblePlot and LinePlot: Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
    }]
    return bubble_line_graph




# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DiasterResponseData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    bar_graph = [
        {
            'data': [
                Bar(
                    y=genre_names,
                    x=genre_counts,
                    orientation = 'h',

                    marker = dict(
                      color = 'rgb(207, 107, 107)'
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        }
    ]

    #Another distribution of : Bubble and Line Graph
    bubble_line_graph = bubble_line_plot_graph(genre_names, genre_counts)
   
    graphs = []
    graphs.append(bar_graph[0])
    graphs.append(bubble_line_graph[0])

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
        
    print(ids)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    print(query)
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    print(classification_labels)
    print(classification_results)

    predicted_tags = { k:v for k, v in classification_results.items() if v != 0}
    print(predicted_tags)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=predicted_tags
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
