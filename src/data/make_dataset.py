import torch
from torchtext.vocab import  Vocab
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import random
import subprocess
import os
import re
import string
import pandas as pd 
from collections import Counter
from transformers import BertTokenizer  


SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

def custom_standardization(input_data):
    lowercase = input_data.lower()
    stripped_html = re.sub("<br />", " ", lowercase)
    return re.sub(f"[{re.escape(string.punctuation)}]", "", stripped_html)

def text_pipeline(x, tokenizer):
    tokenized_text = tokenizer(x, padding=True, truncation=True, return_tensors='pt')
    return tokenized_text['input_ids'].squeeze(), tokenized_text['attention_mask'].squeeze()

def label_pipeline(x):
    return torch.tensor([0]) if x == 'FAKE' else torch.tensor([1])

def build_vocab(data, tokenizer, min_freq=5):
    counter = Counter()
    for (label, line) in data:
        tokenized_output = tokenizer(custom_standardization(line), padding=True, truncation=True, return_tensors='pt')
        input_ids = tokenized_output['input_ids'].squeeze()
        counter.update(input_ids)

    # Remove infrequent words
    filtered_tokens = [token for token, count in counter.items() if count >= min_freq]

    # Create the vocabulary without infrequent words
    vocab = Vocab(Counter(filtered_tokens))
    
    return vocab



class TextClassificationDataset(Dataset):
    def __init__(self, examples, text_pipeline, label_pipeline):
        super(TextClassificationDataset, self).__init__()
        self.examples = examples
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline

    def __getitem__(self, i):
        return (self.text_pipeline(self.examples[i][0]), self.label_pipeline(self.examples[i][1]))

    def __len__(self):
        return len(self.examples)

def collate_fn(batch):
    texts, labels = zip(*batch)
    return {'texts': texts, 'labels': torch.stack(labels)}

if __name__ == '__main__':
    df = get_data()

    # Replace 'text_field' and 'label_field' with 'text' and 'label'
    examples = [(row['text'], row['label']) for _, row in df.iterrows()]
    train_examples, valid_examples = train_test_split(examples, test_size=0.2, random_state=SEED)



# Split the data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Replace 'text_field' and 'label_field' with 'text' and 'label'
train_examples = [(row['text'], row['label']) for _, row in train_df.iterrows()]
valid_examples = [(row['text'], row['label']) for _, row in valid_df.iterrows()]
test_examples = [(row['text'], row['label']) for _, row in test_df.iterrows()]

vocab = build_vocab(train_examples, tokenizer)

train_data = TextClassificationDataset(train_examples, lambda x: text_pipeline(x[0], tokenizer), label_pipeline)
valid_data = TextClassificationDataset(valid_examples, lambda x: text_pipeline(x[0], tokenizer), label_pipeline)
test_data = TextClassificationDataset(test_examples, lambda x: text_pipeline(x[0], tokenizer), label_pipeline)

BATCH_SIZE = 64

train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)



token_frequencies = Counter()
for i in range(5):
    print(f"Example {i+1}:")
    original_text = train_examples[i][0]
    input_ids, attention_mask = tokenizer(custom_standardization(original_text))['input_ids'], tokenizer(custom_standardization(original_text))['attention_mask']
    print("Original text:", original_text)
    print("Tokenized input_ids:", input_ids)
    print("Tokenized attention_mask:", attention_mask)
    print("Label:", train_data[i][1])

    # Update token frequencies for the current example
    tokens = input_ids
    token_frequencies.update(tokens)

    print()

print("Token frequencies:", token_frequencies)
#what is going on here?