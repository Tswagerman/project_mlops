import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class Network(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x = batch.text[0]  # Access the 'text' field from the batch
        y = batch.label.float()  # Access the 'label' field from the batch
        output = self(x)
        loss = nn.CrossEntropyLoss()(output, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
