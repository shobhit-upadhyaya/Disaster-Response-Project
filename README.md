# Disaster Response Pipeline Project

### Project Overview:
    Disaster Response Pipeline Project is a multiclass classification problem, where we are analyzing the disaster data from Figure Eight to build a model for an API that classifies disaster messages.
    
    This project has built on a dataset containing real messages that were sent during disaster events. We have created a machine learning pipeline to categorize these events so that model can send the messages to an appropriate disaster relief agency. 
    

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
