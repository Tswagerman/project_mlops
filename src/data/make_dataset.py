import torch
import os
import pandas as pd
import subprocess
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from collections import Counter


def get_dvc_remote_path(remote_name):
    result = subprocess.run(["dvc", "remote", "default", remote_name], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError(f"Failed to get DVC remote path: {result.stderr.strip()}")


def get_data():
    # Run DVC pull to fetch data from the remote
    os.system("dvc pull -r public-remote")

    # Retrieve the local DVC cache path from the DVC configuration
    dvc_remote_path = get_dvc_remote_path("public-remote")

    # Load the CSV file into a Pandas DataFrame
    csv_file_path = os.path.join(dvc_remote_path, "data/raw/news.csv")  # Adjust the path to your CSV file
    df = pd.read_csv(csv_file_path)

    return df


# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]["text"])
        label = int(self.data.iloc[index]["label"])  # Assuming 'label' is 0 or 1

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def getDatasets():
    df = get_data()

    df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_df, tokenizer)
    test_dataset = CustomDataset(test_df, tokenizer)

    return {"train": train_dataset, "test": test_dataset}


# --------------------------


class CustomTextDataset(Dataset):
    def __init__(self, data, vocab, max_length=512):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]["text"])
        label = int(self.data.iloc[index]["label"])

        # Encode the text using your custom method
        encoded_text = self.encode_text(text)

        return {"input_ids": encoded_text, "label": torch.tensor(label, dtype=torch.long)}

    def encode_text(self, text):
        # Example encoding method: convert each word to an integer based on your custom vocabulary
        # You might need to tokenize the text first, depending on your vocab structure
        tokens = text.split()[: self.max_length]  # Tokenize and truncate
        encoded_text = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Padding
        padding_length = self.max_length - len(encoded_text)
        encoded_text += [self.vocab["<PAD>"]] * padding_length

        return torch.tensor(encoded_text, dtype=torch.long)


# -------------------------


def build_vocab(texts, max_vocab_size):
    # Tokenize texts and build vocabulary
    all_tokens = [token for text in texts for token in text.split()]
    token_freq = Counter(all_tokens)
    vocab = {token: idx for idx, (token, _) in enumerate(token_freq.most_common(max_vocab_size))}
    vocab["<PAD>"] = len(vocab)  # Padding token
    vocab["<UNK>"] = len(vocab)  # Unknown token

    return vocab


def getDatasets_custom_transformer():
    df = get_data()

    df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Build vocabulary from training set
    vocab = build_vocab(train_df["text"], max_vocab_size=10000)  # Adjust the max_vocab_size as needed

    # Create datasets
    train_dataset = CustomTextDataset(train_df, vocab, max_length=512)
    test_dataset = CustomTextDataset(test_df, vocab, max_length=512)

    return {"train": train_dataset, "test": test_dataset}
