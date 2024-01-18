from data.make_dataset import get_data, CustomDataset, getDatasets, CustomTextDataset, getDatasets_custom_transformer
import pandas as pd
from transformers import BertTokenizer


def test_get_data():
    df = get_data()
    assert isinstance(df, pd.DataFrame), "get_data should return a DataFrame"
    assert "text" in df.columns, "DataFrame should have a 'text' column"
    assert "label" in df.columns, "DataFrame should have a 'label' column"


def test_custom_dataset():
    dummy_data = pd.DataFrame({"text": ["This is a sample text."], "label": [0]})
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dummy_dataset = CustomDataset(dummy_data, tokenizer)
    assert len(dummy_dataset) == 1, "CustomDataset should have length 1 for the dummy data"
    sample_item = dummy_dataset[0]
    assert "input_ids" in sample_item, "Sample item should have 'input_ids'"
    assert "attention_mask" in sample_item, "Sample item should have 'attention_mask'"
    assert "label" in sample_item, "Sample item should have 'label'"


def test_get_datasets():
    datasets = getDatasets()
    assert "train" in datasets, "getDatasets should return a dictionary with 'train' dataset"
    assert "test" in datasets, "getDatasets should return a dictionary with 'test' dataset"
    assert isinstance(datasets["train"], CustomDataset), "'train' dataset should be an instance of CustomDataset"
    assert isinstance(datasets["test"], CustomDataset), "'test' dataset should be an instance of CustomDataset"


def test_custom_text_dataset():
    dummy_data = pd.DataFrame({"text": ["This is a sample text."], "label": [0]})
    vocab = {"<PAD>": 0, "<UNK>": 1, "This": 2, "is": 3, "a": 4, "sample": 5, "text.": 6}
    dummy_dataset = CustomTextDataset(dummy_data, vocab)
    assert len(dummy_dataset) == 1, "CustomTextDataset should have length 1 for the dummy data"
    sample_item = dummy_dataset[0]
    assert "input_ids" in sample_item, "Sample item should have 'input_ids'"
    assert "label" in sample_item, "Sample item should have 'label'"


def test_get_datasets_custom_transformer():
    datasets = getDatasets_custom_transformer()
    assert "train" in datasets, "getDatasets_custom_transformer should return a dictionary with 'train' dataset"
    assert "test" in datasets, "getDatasets_custom_transformer should return a dictionary with 'test' dataset"
    assert isinstance(
        datasets["train"], CustomTextDataset
    ), "'train' dataset should be an instance of CustomTextDataset"
    assert isinstance(datasets["test"], CustomTextDataset), "'test' dataset should be an instance of CustomTextDataset"
