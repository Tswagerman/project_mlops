import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from data.lightningDatasetClass import NewsDataset, NewsModule

from data.make_dataset import get_data
from models.model import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the path to the saved data
data_path = 'data/raw/news.csv'


def load_data() -> pd.DataFrame:
    """Load the data.

    Returns:
        Pandas DataFrame containing the processed data.

    """
    # Check if the saved data exists
    if not os.path.exists(data_path):
        # Run make_dataset.py to generate the data
        print('Generating data')
        os.system('python src/data/make_dataset.py')

    print('Loading data from saved location')
    data_df = pd.read_csv(data_path)
    return data_df


def train(df) -> None:
    """Train the model."""
    #batch_size = 64

    #news_dataset = NewsDataset(df)
    #news_module = NewsModule(news_dataset, batch_size)

    #trainer = pl.Trainer(max_epochs=5)  # Adjust max_epochs and gpus based on your setup
    #trainer.fit(news_module, news_module.train_dataloader(), news_module.test_dataloader())
    dataset = NewsDataset(df)
    dataset[0]

if __name__ == '__main__':
    # Load the data
    df = load_data()
    # Train the model
    #print(df)
    train(df)