import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Network(pl.LightningModule):
    def __init__(self, max_words, max_sequence_length):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=max_words, embedding_dim=32)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * max_sequence_length, 64)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy(logits, y.float().view(-1, 1))
        return loss