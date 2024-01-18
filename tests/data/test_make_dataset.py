import unittest
import pytest
from data.make_dataset import get_data, CustomDataset, getDatasets, CustomTextDataset, getDatasets_custom_transformer
import pandas as pd
from transformers import BertTokenizer  

class TestDataFunctions(unittest.TestCase):
    def test_get_data(self):
        df = get_data()
        self.assertTrue(isinstance(df, pd.DataFrame), "get_data should return a DataFrame")
        self.assertIn('text', df.columns, "DataFrame should have a 'text' column")
        self.assertIn('label', df.columns, "DataFrame should have a 'label' column")

    def test_custom_dataset(self):
        dummy_data = pd.DataFrame({'text': ['This is a sample text.'], 'label': [0]})
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        dummy_dataset = CustomDataset(dummy_data, tokenizer)
        self.assertEqual(len(dummy_dataset), 1, "CustomDataset should have length 1 for the dummy data")
        sample_item = dummy_dataset[0]
        self.assertIn('input_ids', sample_item, "Sample item should have 'input_ids'")
        self.assertIn('attention_mask', sample_item, "Sample item should have 'attention_mask'")
        self.assertIn('label', sample_item, "Sample item should have 'label'")

    def test_get_datasets(self):
        datasets = getDatasets()
        self.assertIn('train', datasets, "getDatasets should return a dictionary with 'train' dataset")
        self.assertIn('test', datasets, "getDatasets should return a dictionary with 'test' dataset")
        self.assertIsInstance(datasets['train'], CustomDataset, "'train' dataset should be an instance of CustomDataset")
        self.assertIsInstance(datasets['test'], CustomDataset, "'test' dataset should be an instance of CustomDataset")

    def test_custom_text_dataset(self):
        dummy_data = pd.DataFrame({'text': ['This is a sample text.'], 'label': [0]})
        vocab = {'<PAD>': 0, '<UNK>': 1, 'This': 2, 'is': 3, 'a': 4, 'sample': 5, 'text.': 6}
        dummy_dataset = CustomTextDataset(dummy_data, vocab)
        self.assertEqual(len(dummy_dataset), 1, "CustomTextDataset should have length 1 for the dummy data")
        sample_item = dummy_dataset[0]
        self.assertIn('input_ids', sample_item, "Sample item should have 'input_ids'")
        self.assertIn('label', sample_item, "Sample item should have 'label'")

    def test_get_datasets_custom_transformer(self):
        datasets = getDatasets_custom_transformer()
        self.assertIn('train', datasets, "getDatasets_custom_transformer should return a dictionary with 'train' dataset")
        self.assertIn('test', datasets, "getDatasets_custom_transformer should return a dictionary with 'test' dataset")
        self.assertIsInstance(datasets['train'], CustomTextDataset, "'train' dataset should be an instance of CustomTextDataset")
        self.assertIsInstance(datasets['test'], CustomTextDataset, "'test' dataset should be an instance of CustomTextDataset")

if _name_ == '_main_':
    unittest.main()