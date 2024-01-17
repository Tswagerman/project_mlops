import subprocess
import os
import pandas as pd 
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def get_dvc_remote_path(remote_name) -> str:
    result = subprocess.run(['dvc', 'remote', 'default', remote_name], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError(f"Failed to get DVC remote path: {result.stderr.strip()}")

def tokenize_and_save(df, processed_data_path) -> None:
    nltk.download('punkt')
    nltk.download('stopwords')

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    print('Processing data...')
    # Conversion of text before tokenization
    print('Converting text to lowercase...')
    df['text'] = df['text'].astype(str).str.lower()

    # Remove non-alphanumeric characters with spaces
    print('Removing non-alphanumeric characters...')
    df['text'] = df['text'].apply(lambda x: re.sub(r'\W', ' ', x))

    # Tokenization and stemming
    print('Tokenizing and stemming...')
    df['text'] = df['text'].apply(lambda x: [stemmer.stem(word) for word in word_tokenize(x)])
    
    # Remove stop words
    print('Removing stop words...')
    df['text'] = df['text'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
    print(df['text'])
    # Join tokens
    #print('Joining tokens...')
    #df['text'] = df['text'].apply(lambda x: ' '.join(x))
    print('Data processing complete')
    # Remove empty rows
    df = df[df['text'].map(len) > 0]
    # Save processed data
    print('Saving processed data...')
    df.to_csv(processed_data_path, index=False)
    print('Processed data saved')

def get_data() -> pd.DataFrame:
    # Run DVC pull to fetch data from the remote
    if not os.path.exists('data/raw/news.csv'):
        os.system('dvc pull -r public-remote')
    else:
        print('Raw data already exists')

    # Retrieve the local DVC cache path from the DVC configuration
    dvc_remote_path = get_dvc_remote_path('public-remote')

    # Load the CSV file into a Pandas DataFrame
    csv_file_path = os.path.join(dvc_remote_path, 'data/raw/news.csv')  # Adjust the path to your CSV file
    df = pd.read_csv(csv_file_path)
    print(df.columns)
    
    df['text'] = df['text'].astype(str)
    df.dropna(inplace=True)
    df['label'] = df['label'].replace('REAL', '0')
    df['label'] = df['label'].replace('FAKE', '1')
    # Changing the datatype of label column to int32
    # Convert labels to a list of integers
    df['label'] = df['label'].astype('int32').tolist()
    print("labels = ", df['label'].values , "label type = ", type(df['label'].values ))

    df.drop(columns=['title'], inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)

    processed_data_path = 'data/processed/processed_data.csv'

    # Tokenize and save if processed data doesn't exist
    if not os.path.exists(processed_data_path):
        print('Processed data does not exist')
        tokenize_and_save(df, processed_data_path)
    else:
        print('Processed data already exists')

    return df

if __name__ == '__main__':
    get_data()
