import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb
import sys
from datetime import datetime

from csv_dataset import CSVDataset
from classifier import FCNStateSelector, CNNStateSelector, LSTMStateSelector
from config import TrainConfig

cfg = TrainConfig()

train_dataset = CSVDataset(cfg.train_csv_file, cfg.csv_input_col, cfg.csv_label_col, -999)
exit(0)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

val_dataset = CSVDataset(cfg.val_csv_file, cfg.csv_input_col, cfg.csv_label_col)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

# Get the first batch of the training data
train_inputs, train_labels = next(iter(train_dataloader))

# Check the batch size
assert len(train_inputs) == cfg.batch_size, f"Expected batch size {cfg.batch_size}, but got {len(train_inputs)}"

# Check the shape of the inputs and labels
print(f"Shape of train_inputs: {train_inputs.shape}")
print(f"Shape of train_labels: {train_labels.shape}")

# Check the data type of the inputs and labels
print(f"Data type of train_inputs: {train_inputs.dtype}")
print(f"Data type of train_labels: {train_labels.dtype}")

# Might wanna run this on Colab so I don't have to keep loading the dataset and I can use my compute units
now = datetime.now()
time_string = now.strftime("%m-%d-%H-%M")
wandb.init(project="endstate-selector")
wandb.run.name = f"{time_string}_endstate-selector"
print(f"Run name: {wandb.run.name}")
print(f"Train CSV file: {cfg.train_csv_file}")
print(f"Val CSV file: {cfg.val_csv_file}")

# Initialize and train your model (example)
model = FCNStateSelector(cfg.input_size, cfg.output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

best_val_loss = float('inf')

model_name = f"models/endstate-selector_{time_string}.pth"

for epoch in range(cfg.num_epochs):
    
    model.train()
    running_loss = 0.0
    for batch, (train_inputs, train_labels) in enumerate(train_dataloader):
        # print(train_inputs.shape)
        # print(train_labels.shape)

        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
          
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch, (val_inputs, val_labels) in enumerate(val_dataloader):
            outputs = model(val_inputs)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()
    
    wandb.log({"val_loss": val_loss})    
    wandb.log({"train_loss": running_loss})
    # print(f"Validation loss: {val_loss / len(val_dataloader)}")
    print(f"Epoch {epoch + 1}, train_loss: {round(running_loss / len(train_dataloader),3)}, val_loss: {round(val_loss / len(val_dataloader),3)}")

    # Save the model if it has the best validation loss so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_name)
        # print(f"Model saved as {model_name}")

# Save the trained model
# torch.save(model.state_dict(), "models/final_model.pth")

# Convert to TorchScript using tracing
# example_input = torch.randn(1, 10)
# traced_script_module = torch.jit.trace(model, example_input)
# traced_script_module.save("simple_model.pt")