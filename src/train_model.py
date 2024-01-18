import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models.model import FakeRealClassifier, TextTransformer
from data.make_dataset import getDatasets

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

# Initialization of Hydra
@hydra.main(
    config_path="models/config/Bert",
    config_name="default_config.yaml",
    version_base=None,
)

def train(config, test=False):
    if not config:
        raise ValueError("Configuration dictionary should not be empty!")

    if config['num_workers'] < 0:
        raise ValueError("Number of workers cannot be negative")

    if config['n_epochs'] <= 0:
        raise ValueError("Number of epochs cannot be zero or negative")

    print(f"Configuration: \n{OmegaConf.to_yaml(config)}")

    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Initialize wandb
    os.environ["WANDB_API_KEY"] = config['wandbAPI']
    wandb.init(project="mlops", entity="team_mlops7")
    

    current_start_method = mp.get_start_method()
    
    if current_start_method != 'spawn':
        mp.set_start_method('spawn')
        
    datasets = getDatasets()

    train_dataloader = DataLoader(
        datasets["train"], 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    test_dataloader = DataLoader(
        datasets["test"], 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )

    # Instantiate the model
    model = FakeRealClassifier()
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(train_dataloader) * config['scheduler_step']
    )
    
    # Initialize the best validation loss to a high value
    best_loss = float('inf')
    # Initialize mixed-precision training
    scaler = GradScaler()
    
        # Training loop
    for epoch in range(config['n_epochs']):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        # Progress bar for training
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config["n_epochs"]}, Training') as train_pbar:
            for step, batch in enumerate(train_pbar):
                # Prepare batch data
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                # Mixed-precision training
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(logits, labels)

                scaler.scale(loss).backward()
                if (step + 1) % config['accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                scheduler.step()
                total_train_loss += loss.item()
                total_samples += labels.size(0)

                average_train_loss = total_train_loss / total_samples
                train_pbar.set_postfix({'Loss': average_train_loss})

                # Log training metrics
                wandb.log({"epoch": epoch + 1, "train_loss": average_train_loss})

        # Validation
        model.eval()
        total_correct = 0
        total_samples = 0
        total_val_loss = 0.0

        # Progress bar for validation
        with tqdm(test_dataloader, desc=f'Epoch {epoch + 1}/{config["n_epochs"]}, Validation') as val_pbar:
            with torch.no_grad():
                for batch in val_pbar:
                    # Prepare batch data
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
                    
                    # Log validation metrics
                    wandb.log({"epoch": epoch + 1, "val_loss": average_val_loss, "val_accuracy": accuracy})
        
        # Model saving based on validation loss
        if abs(average_val_loss-average_train_loss) < best_loss and not test:
            best_loss = abs(average_val_loss - average_train_loss)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Best model saved!")

    print("Training finished.")



if __name__ == '__main__':
    train()