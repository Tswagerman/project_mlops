import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from data.make_dataset import get_data
from models.model import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def load_data() -> pd.DataFrame:
    """Load the data.

    Returns:
        Pandas DataFrame containing the processed data.

    """
    # Define the path to the saved data
    saved_data_path = 'data/processed/processed_data.csv'

    # Check if the saved data exists
    if not os.path.exists(saved_data_path):
        # Run make_dataset.py to generate the data
        print('Generating data')
        os.system('python src/data/make_dataset.py')

    print('Loading data from saved location')
    data_df = pd.read_csv(saved_data_path)
    return data_df

def train(df) -> None:
    """Train the model."""
    # Divide data
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Ensure 'text' column is processed correctly
    train_df['text'] = train_df['text'].apply(lambda x: x.split())  # adjust this based on your actual tokenization
    test_df['text'] = test_df['text'].apply(lambda x: x.split())    # adjust this based on your actual tokenization

    trainloader = DataLoader(train_df, batch_size=64, shuffle=True)
    testloader = DataLoader(test_df, batch_size=64, shuffle=True)

    # Initialize model
    in_features = len(train_df.columns) - 1  # Exclude the 'label' column
    out_features = 2  # Binary classification
    model = Network(in_features, out_features)

    # Convert DataFrame to NumPy array with the correct data type
    X_train = torch.FloatTensor(train_df.drop(columns=['label']).values.astype(np.float32))
    y_train = torch.LongTensor(train_df['label'].values)

    X_test = torch.FloatTensor(test_df.drop(columns=['label']).values.astype(np.float32))
    y_test = torch.LongTensor(test_df['label'].values)

    trainer = pl.Trainer(max_epochs=5)  # Adjust max_epochs and gpus based on your setup
    trainer.fit(model, DataLoader(X_train, y_train, batch_size=64, shuffle=True),
                DataLoader(X_test, y_test, batch_size=64, shuffle=True))

if __name__ == '__main__':
    # Load the data
    data_df = load_data()
    # Train the model
    train(data_df)
