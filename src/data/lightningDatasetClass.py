import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pandas import read_csv
from models.model import Network
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')  # tokenizer
MAX_LENGTH = 150
pad_token = '<pad>'
unk_token = '<unk>'
VOCAB_SIZE = 1500


class NewsDataset(Dataset):
    def __init__(self, df, test_dataset=False):
        super().__init__()
        self.df = df
        self.test_dataset = test_dataset
        self.text_tokens = df.text.map(tokenizer)
        self.vocab = build_vocab_from_iterator(self.text_tokens,
                                               specials=[unk_token, pad_token],
                                               max_tokens=VOCAB_SIZE)

    def __len__(self):
        return len(self.df)

    def pad_tokens(self, tokens):
        if len(tokens) > MAX_LENGTH:
            return tokens[:MAX_LENGTH]
        else:
            return tokens + [pad_token] * (MAX_LENGTH - len(tokens))

    def __getitem__(self, idx):
        text = self.df.text.values[idx]
        text_tokens = self.pad_tokens(tokenizer(text))
        input = torch.tensor(self.vocab.lookup_indices(text_tokens))
        if self.test_dataset:
            target = torch.tensor([0, 0, 0, 0, 0, 0]).float()  # for the test dataset, make the target all zeros
        else:
            target = torch.tensor(self.df.label.values[idx]).float()
        return input, target


class NewsModule(pl.LightningModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data = read_csv(data_path)
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = Network(10000, 100, 2)
        self.train_data, self.test_data = self.load_data()

    def load_data(self):
        train_data, test_data = train_test_split(self.data, test_size=0.2)
        return train_data, test_data

    def train_dataloader(self):
        return DataLoader(NewsDataset(self.train_data), batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(NewsDataset(self.test_data, test_dataset=True), batch_size=self.batch_size, shuffle=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        # Define optimizer and scheduler here if needed
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)