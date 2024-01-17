import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
from models.model import FakeRealClassifier
from tqdm import tqdm 

from data.make_dataset import getDatasets


datasets = getDatasets()
train_dataloader = DataLoader(datasets["train"], batch_size=8, shuffle=True)
test_dataloader = DataLoader(datasets["test"], batch_size=8, shuffle=False)

# Instantiate the model
model = FakeRealClassifier()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 10)

# Training loop
num_epochs = 3

# Initialize the best validation loss to a high value
best_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0.0
    total_samples = 0

    # Use tqdm for progress bar
    with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}, Training') as train_pbar:
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_samples += labels.size(0)

            average_train_loss = total_train_loss / total_samples
            train_pbar.set_postfix({'Loss': average_train_loss})

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    total_val_loss = 0.0

    # Use tqdm for progress bar
    with tqdm(test_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}, Validation') as val_pbar:
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_val_loss += loss.item()

                _, predictions = torch.max(logits, 1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                average_val_loss = total_val_loss / total_samples
                accuracy = total_correct / total_samples
                val_pbar.set_postfix({'Loss': average_val_loss, 'Accuracy': accuracy})
    
    # Save the model if it has the best validation loss
    if average_val_loss < best_loss:
        best_loss = average_val_loss
        torch.save(model.state_dict(), 'models/best_model.pth')
        print("Best model saved!")

print("Training finished.")

