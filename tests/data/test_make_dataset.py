import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from unittest.mock import Mock, patch
import pytest
from src.data.make_dataset import getDatasets, CustomDataset, get_dvc_remote_path, get_data

@pytest.fixture
def mock_subprocess_run():
    with patch('subprocess.run') as mock:
        yield mock

@pytest.fixture
def mock_tokenizer():
    return Mock()

def test_get_dvc_remote_path_success(mock_subprocess_run):
    mock_subprocess_run.return_value.returncode = 0
    mock_subprocess_run.return_value.stdout = 'your_dvc_remote_path\n'
    result = get_dvc_remote_path('your_remote_name')
    assert result == 'your_dvc_remote_path'

def test_get_dvc_remote_path_failure(mock_subprocess_run):
    mock_subprocess_run.return_value.returncode = 1
    mock_subprocess_run.return_value.stderr = 'Error fetching DVC remote path\n'
    with pytest.raises(RuntimeError):
        get_dvc_remote_path('your_remote_name')

def test_custom_dataset(mock_tokenizer):
    # Mock data for the dataset
    data = pd.DataFrame({
        'text': ['text1', 'text2'],
        'label': [0, 1]
    })

    # Create an instance of the CustomDataset
    dataset = CustomDataset(data, mock_tokenizer)

    # Check if __len__ returns the correct length
    assert len(dataset) == 2

    # Check if __getitem__ returns the correct format
    sample = dataset[0]
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'label' in sample
    assert isinstance(sample['input_ids'], torch.Tensor)
    assert isinstance(sample['attention_mask'], torch.Tensor)
    assert isinstance(sample['label'], torch.Tensor)

def test_get_datasets(mock_tokenizer):
    # Mock data for getDatasets function
    data = pd.DataFrame({
        'text': ['text1', 'text2'],
        'label': ['FAKE', 'REAL']
    })
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Mock the get_data function
    with patch('your_script_file.get_data', return_value=data):
        datasets = getDatasets()

    # Check if the returned datasets are instances of CustomDataset
    assert isinstance(datasets['train'], CustomDataset)
    assert isinstance(datasets['test'], CustomDataset)
