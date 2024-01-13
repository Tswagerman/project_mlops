import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from src.data.make_dataset import DatasetNews
from model import TextTransformer, Bert  # Import your TextTransformer class
import hydra
from omegaconf import DictConfig
import wandb

# Function to train for one epoch
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

# Main training function
@hydra.main(
    config_path="config/Bert",
    config_name="default_config.yaml",
    version_base=None,
)

def train(cfg: DictConfig):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Unpack experiment specific params
    hparams = config["_group_"]  # wtf is this __group__ ?
    
    # Wandb initialization
    wandb.init(project=f"BERT", entity="dvc") # ENTITY ?
    wandb.config = {
        "learning_rate": lr,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "vit": hparams.vit,
        "text_transf": hparams.text_transf,
    }

    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')

    train_dataset = DatasetNews(cfg.data.train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    model = TextTransformer(num_heads=cfg.model.num_heads, 
                            num_blocks=cfg.model.num_blocks, 
                            embed_dims=cfg.model.embed_dims, 
                            vocab_size=cfg.model.vocab_size, 
                            max_seq_len=cfg.model.max_seq_len, 
                            num_classes=2, 
                            dropout=cfg.model.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    for epoch in range(cfg.train.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{cfg.train.num_epochs}], Loss: {train_loss:.4f}")

        # Log metrics to wandb
        wandb.log({"epoch": epoch, "loss": train_loss})

        # Save model, additional logging, etc.

if __name__ == "__main__":
    train()