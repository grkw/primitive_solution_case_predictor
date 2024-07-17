import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb

from csv_dataset import CSVDataset
from classifier import FCNStateSelector, CNNStateSelector, LSTMStateSelector
from config import TestConfig

cfg = TestConfig()

# Load the test data
test_dataset = CSVDataset(cfg.test_csv_file, cfg.csv_input_col, cfg.csv_label_col)  # replace with your test csv file
test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

# Load the best model
best_model = FCNStateSelector(cfg.input_size, cfg.output_size)
best_model.load_state_dict(torch.load(cfg.best_model_path))

# Evaluate the model
criterion = nn.CrossEntropyLoss()
best_model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for batch, (test_inputs, test_labels) in enumerate(test_dataloader):
        outputs = best_model(test_inputs)

        # Calculate loss
        loss = criterion(outputs, test_labels)
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
        print('Predicted', predicted)
        print('Truth', test_labels)

print('Total # of test examples: ', total)
print('# of correctly predicted examples: ', correct)

print(f'Test Loss: {round(test_loss / len(test_dataloader),3)}')
print(f'Test Accuracy: {round(correct / total * 100,3)}%')
print(f'Guessing Accuracy: {round(1 / cfg.num_vf_choices * 100,3)}%')