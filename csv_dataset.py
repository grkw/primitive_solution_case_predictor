import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CSVDataset(Dataset):
    def __init__(self, csv_files, input_col, label_col, num_sample, transform=None):
        
        # Merge the data from all the CSV files into a single DataFrame
        self.data = pd.concat([pd.read_csv(f) for f in csv_files])
        self.data[12:15] = self.data[12:15] - self.data[0:3]
        self.data[15:18] = self.data[15:18] - self.data[0:3]
        self.data = self.data.drop(self.data.columns[[*range(0, 12), *range(19, 22)]], axis=1)
        # automatically treats the first row as the header and does not include it in the data
        
        print("Data shape: ", self.data.shape)
        self.transform = transform

        self.data = self.data.groupby('v_desc0').sample(n=num_sample, replace=True)
        print("Data shape after sampling: ", self.data.shape)

        self.le = LabelEncoder()
        self.data.iloc[:, label_col] = self.le.fit_transform(self.data.iloc[:, label_col])
        print("LabelEncoder classes shape: ", self.le.classes_.shape)

        labels = self.data.iloc[:, label_col]
        label_distribution = np.bincount(labels)
        # Pair each class with its frequency
        class_distribution = zip(self.le.classes_, label_distribution)
        # Print the classes and their frequencies
        for class_, frequency in class_distribution:
            print(f"Class: {class_}, Frequency: {frequency}")

        self.label_col = label_col
        self.input_col = input_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.data.iloc[idx, :self.input_col]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        
        outputs = self.data.iloc[idx, self.label_col]
        outputs = torch.tensor(outputs, dtype=torch.long)

        # if self.transform:
        #     sample = self.transform(sample)

        return inputs, outputs
