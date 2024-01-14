import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import subprocess
import os
import pandas as pd 
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy, f1_score, auroc
import multiprocessing



def get_dvc_remote_path(remote_name):
    result = subprocess.run(['dvc', 'remote', 'default', remote_name], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError(f"Failed to get DVC remote path: {result.stderr.strip()}")

def get_data():
    # Run DVC pull to fetch data from the remote
    os.system('dvc pull -r public-remote')

    # Retrieve the local DVC cache path from the DVC configuration
    dvc_remote_path = get_dvc_remote_path('public-remote')

    # Load the CSV file into a Pandas DataFrame
    csv_file_path = os.path.join(dvc_remote_path, 'data/raw/news.csv')  # Adjust the path to your CSV file
    df = pd.read_csv(csv_file_path)

    return df

class ToxicCommentsDataset(Dataset):

  def __init__(

    self,

    data: pd.DataFrame,

    tokenizer: BertTokenizer,

    max_token_len: int = 512,

    label_cols: list = None

  ):

    self.tokenizer = tokenizer

    self.data = data

    self.labels = label_cols

    self.max_token_len = max_token_len

  def __len__(self):

    return len(self.data)

  def __getitem__(self, index: int):

    data_row = self.data.iloc[index]

    comment_text = data_row.text

    labels = data_row[self.labels]


    encoding = self.tokenizer.encode_plus(

      comment_text,

      add_special_tokens=True,

      max_length=self.max_token_len,

      return_token_type_ids=False,

      padding="max_length",

      truncation=True,

      return_attention_mask=True,

      return_tensors='pt',

    )

    return dict(

      comment_text=comment_text,

      input_ids=encoding["input_ids"].flatten(),

      attention_mask=encoding["attention_mask"].flatten(),

      labels=torch.FloatTensor(labels)

    )
    
class ToxicCommentDataModule(pl.LightningDataModule):

  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128, labels: list=None):

    super().__init__()

    self.batch_size = batch_size

    self.train_df = train_df

    self.test_df = test_df

    self.tokenizer = tokenizer

    self.max_token_len = max_token_len

    self.labels = labels

  def setup(self, stage=None):

    self.train_dataset = ToxicCommentsDataset(

      self.train_df,

      self.tokenizer,

      self.max_token_len,

      self.labels

    )

    self.test_dataset = ToxicCommentsDataset(

      self.test_df,

      self.tokenizer,

      self.max_token_len,
     
      self.labels
    )

  def train_dataloader(self):

    return DataLoader(

      self.train_dataset,

      batch_size=self.batch_size,

      shuffle=True,

      num_workers=4

    )

  def val_dataloader(self):

    return DataLoader(

      self.test_dataset,

      batch_size=self.batch_size,
      persistent_workers=True,

      num_workers=4

    )

  def test_dataloader(self):

    return DataLoader(

      self.test_dataset,

      batch_size=self.batch_size,

      num_workers=4

    )

class ToxicCommentTagger(pl.LightningModule):

  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None , labels=None):

    super().__init__()

    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=True)

    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.label_cols = labels

    self.n_training_steps = n_training_steps

    self.training_step_outputs = []

    self.n_warmup_steps = n_warmup_steps

    self.criterion = nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels=None):

    output = self.bert(input_ids, attention_mask=attention_mask)

    output = self.classifier(output.pooler_output)

    output = torch.sigmoid(output)

    loss = 0

    if labels is not None:

        loss = self.criterion(output, labels)

    return loss, output

  def training_step(self, batch, batch_idx):

    input_ids = batch["input_ids"]

    attention_mask = batch["attention_mask"]

    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)

    self.log("train_loss", loss, prog_bar=True, logger=True)

    self.training_step_outputs.append(loss)

    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):

    input_ids = batch["input_ids"]

    attention_mask = batch["attention_mask"]

    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)

    self.log("val_loss", loss, prog_bar=True, logger=True)

    return loss

  def test_step(self, batch, batch_idx):

    input_ids = batch["input_ids"]

    attention_mask = batch["attention_mask"]

    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)

    self.log("test_loss", loss, prog_bar=True, logger=True)

    return loss

  def on_train_epoch_end(self):

    labels = []

    predictions = []

    for output in self.training_step_outputs:

      for out_labels in output["labels"].detach().cpu():

        labels.append(out_labels)

      for out_predictions in output["predictions"].detach().cpu():

        predictions.append(out_predictions)

    labels = torch.stack(labels).int()

    predictions = torch.stack(predictions)

    for i, name in enumerate(self.label_cols):

      class_roc_auc = auroc(predictions[:, i], labels[:, i])

      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(

      optimizer,

      num_warmup_steps=self.n_warmup_steps,

      num_training_steps=self.n_training_steps

    )

    return dict(

      optimizer=optimizer,

      lr_scheduler=dict(

        scheduler=scheduler,

        interval='step'

      )

    )
  
if __name__ == '__main__':
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    
    device = torch.device(dev)  
    bert = BertModel.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    multiprocessing.freeze_support()
    df = pd.read_csv("data/raw/news.csv")
    
    # Convert 'news_type' column into dummy columns
    df_labels = pd.get_dummies(df['label'])

    # Rename the columns as per your requirement
    df_labels.rename(columns={'FAKE': 'FAKE', 'REAL': 'REAL'}, inplace=True)
    df = pd.concat([df, df_labels], axis=1)
    df['REAL'] = df['REAL'].astype(int)
    df['FAKE'] = df['FAKE'].astype(int)
    # Replace 'text_field' and 'label_field' with 'text' and 'label'
    examples = [(row['text'], row['label']) for _, row in df.iterrows()]
    train_examples, valid_examples = train_test_split(examples, test_size=0.2, random_state=42)
    LABEL_COLUMNS = df.columns.tolist()[4:]
    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dataset = ToxicCommentsDataset(

    train_df,

    tokenizer,

    max_token_len=512,

    label_cols = LABEL_COLUMNS

    )

    N_EPOCHS = 10

    BATCH_SIZE = 12

    data_module = ToxicCommentDataModule(

    train_df,

    valid_df,

    tokenizer,

    batch_size=BATCH_SIZE,

    max_token_len=512,

    labels = LABEL_COLUMNS

    )

    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5

    # model = NewsBert(bert)
    model = ToxicCommentTagger(

    n_classes=len(LABEL_COLUMNS),

    n_warmup_steps=warmup_steps,

    n_training_steps=total_training_steps,

    labels = LABEL_COLUMNS
    )
    model.to(device)


    checkpoint_callback = ModelCheckpoint(

    dirpath="checkpoints",

    filename="best-checkpoint",

    save_top_k=1,

    verbose=True,

    monitor="val_loss",

    mode="min"

    )

    logger = TensorBoardLogger("lightning_logs", name="toxic-comments")

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = pl.Trainer(

      logger=logger,

      callbacks=[early_stopping_callback,checkpoint_callback],

      max_epochs=N_EPOCHS

    )

    trainer.fit(model, data_module)
