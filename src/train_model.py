import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from data.make_dataset import getDatasets
import wandb
import random 

wandb.init(
    # set the wandb project where this run will be logged
    project="mlops",
)
datasets = getDatasets()
train_dataloader = DataLoader(datasets["train"], batch_size=8, shuffle=True)
test_dataloader = DataLoader(datasets["test"], batch_size=8, shuffle=False)

# Define the model
class FakeRealClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-cased', num_labels=2):
        super(FakeRealClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        return output.logits

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
    model.train()
    total_train_loss = 0.0
    total_samples = 0

    for batch in train_dataloader:
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
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {average_train_loss:.4f}')

    # Log training metrics
    wandb.log({"epoch": epoch + 1, "train_loss": average_train_loss})
        
    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
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
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

        # Log validation metrics
        wandb.log({"epoch": epoch + 1, "val_loss": average_val_loss, "val_accuracy": accuracy})


        # Save the model if it has the best validation loss
        if abs(average_val_loss - average_train_loss) < best_loss:
            best_loss = abs(average_val_loss - average_train_loss)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Best model saved!")

print("Training finished.")


