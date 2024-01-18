import pytest
import sys
import time
import threading
import types
from train_model import train

          

# Test if the training function raises an exception with invalid input
def test_train_with_empty_input():
    config = {}

    with pytest.raises(Exception, match="Configuration dictionary should not be empty!"):
        train(config)
        
# Test if the training function raises an exception with zero or negative epochs
def test_train_with_zero_epochs():
    config = {"batch_size": 16, "lr": 0.001, "n_epochs": 0, "scheduler_step": 100, "accumulation_steps": 2, "num_workers": 4, "wandbAPI": "80ea83cfdd37762205e46cf8ec53ce49fb040e26"}

    with pytest.raises(Exception, match="Number of epochs cannot be zero or negative"):
        train(config)

# Test if the training function raises an exception with negative num_workers
def test_train_with_negative_workers():
    config = {"batch_size": 16, "lr": 0.001, "n_epochs": 0, "scheduler_step": 100, "accumulation_steps": 2, "num_workers": -1, "wandbAPI": "80ea83cfdd37762205e46cf8ec53ce49fb040e26"}

    with pytest.raises(Exception, match="Number of workers cannot be negative"):
        train(config)


# Test if the training function runs successfully with different batch sizes
def test_train_with_different_batch_sizes():
    batch_sizes = [8]
    for batch_size in batch_sizes:
        config = {"batch_size": batch_size, "lr": 0.001, "n_epochs": 1, "scheduler_step": 100, "accumulation_steps": 2, "num_workers": 4}
        try:
            train(config)
        except Exception as e:
            pytest.fail("Config file thrown an exception it shouldn't!") 
        else:
            assert True == True

# Test if the training function runs successfully with different learning rates
def test_train_with_different_learning_rates(capsys):
    learning_rates = [0.001, 0.01, 0.1]
    for lr in learning_rates:
        config = {"batch_size": 16, "lr": lr, "n_epochs": 1, "scheduler_step": 100, "accumulation_steps": 2, "num_workers": 4, "wandbAPI": "80ea83cfdd37762205e46cf8ec53ce49fb040e26"}
        try:
            with capsys.disabled():
                train(config)
        except Exception as e:
            pytest.fail("Config file thrown an exception it shouldn't!")
        

        time.sleep(15)


# Add more tests to cover different scenarios, such as:
# - Testing the behavior with a different model architecture (if applicable)
# - Test the loading of a saved model
# - Test obtaining a dataset
# - Test 
# ...

if __name__ == '__main__':
    pytest.main(['-v', __file__])
