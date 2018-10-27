import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
  """
    Input: messages_filepath , categories_filepath
    Return: dataframe

    Desc:
      load_data function helps loading the data from the file and returns df.
  
  """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="outer", on="id")
    return df


def seperate_category_columns(df):
  """
    Input: dataframe
    Returns: dataframe of categories

    Desc:
      Split `categories` into separate category columns.
      Split the values in the `categories` column on the `;` character so that each value becomes a separate column. 
      You'll find pandas.Series.str.split.html very helpful! Make sure to set `expand=True`.
      Use the first row of categories dataframe to create column names for the categories data.
      Rename columns of `categories` with new column names.
  """
  df_categories = df.categories.str.split(';', expand=True)

  # select the first row of the categories dataframe
  row = df_categories.iloc[0]

  # use this row to extract a list of new column names for categories.
  # one way is to apply a lambda function that takes everything 
  # up to the second to last character of each string with slicing
  category_colnames = row.apply(lambda x: x[:-2]).values
  #print(category_colnames)

  # rename the columns of `categories`
  df_categories.columns = category_colnames

  # Convert category values to just numbers 0 or 1.
  # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
  #  For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
  # - You can perform indexing-with-str, like indexing, by including `.str` after the Series. 
  # You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

  for column in category_colnames:
      # set each value to be the last character of the string
      df_categories[column] = df_categories[column].str[-1]
      
      # convert column from string to numeric
      df_categories[column] = df_categories[column].astype('int')
  
  #print(df_categories.head())

  return df_categories 


def clean_data(df):
  """
    Input: dataframe
    Return: transformed clean dataframe

    Desc:

      Split `categories` into separate category columns.
      Concats the categories into to input dataframe
      Removes the duplicate rows of dataframe

  """
  transformed_categories = seperate_category_columns(df)
  print(transformed_categories.shape)

  # Replace categories column in df with new category columns.
  # -> Drop the categories column from the df dataframe since it is no longer needed.
  # -> Concatenate df and categories data frames.
  df.drop('categories', axis=1, inplace=True)
  df = pd.concat([df, transformed_categories], axis=1)
  print(df.shape)
  print(df.columns)


  # Remove duplicates.
  # - Check how many duplicates are in this dataset.
  # - Drop the duplicates.
  # - Confirm duplicates were removed.
  print("Number of duplicates {}".format(sum(df.duplicated('id'))))
  df.drop_duplicates('id',inplace=True)
  print("Number of duplicates {}".format(sum(df.duplicated('id'))))


  print(df.shape)

  return df


def save_data(df, database_filename):
  """
    Input:  dataframe
    Return: None

    Desc:
      Saves the input dataframe into database table. 
      You can do this with pandas.DataFrame.to_sql method combined with the SQLAlchemy library. 
      Remember to import SQLAlchemy's `create_engine`
  """
  engine = create_engine('sqlite:///'+database_filename)
  df.to_sql('DiasterResponseData', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()