from tests import _PATH_DATA

def test_data():
    dataset = _PATH_DATA/'processed'/'news.csv'
    assert len(dataset) == N_train for training and N_test for test
    assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    assert that all labels are represented
