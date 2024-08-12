import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CSVDataset(Dataset):
    def __init__(self, csv_files, input_col, label_cols, num_sample, transform=None):

        print("Loading data from CSV files")
        print("CSV files: ", csv_files)
        
        # Merge the data from all the CSV files into a single DataFrame. Automatically treats the first row as the header and does not include it in the data
        self.data = pd.concat([pd.read_csv(f) for f in csv_files])

        # Subtract to get the relative position #TODO: Is this a good idea? Should I do a similar thing for velocity, accel?
        # self.data[12:15] = self.data[12:15] - self.data[0:3]
        # self.data[15:18] = self.data[15:18] - self.data[0:3]

        print("Original data shape: ", self.data.shape)

        # Drop rows with invalid input
        self.data = self.data[self.data['ruckig_solution_x'] != 'INVALID_INPUT']
        print("Data shape after removing invalids: ", self.data.shape)
        self.transform = transform
        # self.data = self.data.groupby('ruckig_solution_x').sample(n=num_sample, replace=True)
        # print("Data shape after sampling: ", self.data.shape)

        for col in label_cols:
            self.data[col] = self.data[col].replace('_UDUD|_UDDU', '', regex=True)

        print("\nData distributions by class")
        self.le = LabelEncoder()
        for col in label_cols:
            print(f"\n{col}")

            self.data[col] = self.le.fit_transform(self.data[col])
            print(f"Class labels shape: ", self.le.classes_.shape)

            label_distribution = (np.bincount(self.data[col])/self.data.shape[0]*100).round(2)
            # Print the classes and their frequencies
            class_distribution = list(zip(self.le.classes_, label_distribution))
            
            class_distribution_df = pd.DataFrame(class_distribution, columns=['Class', 'Frequency (%)'])
            class_distribution_df = class_distribution_df.sort_values(by='Frequency (%)', ascending=False)
            print(class_distribution_df)
            class_distribution_df.to_csv('dataset_distr.csv', index=False)

        self.label_cols = label_cols
        self.input_col = input_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.data.iloc[idx, :self.input_col]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        
        outputs = [torch.tensor(self.data.iloc[idx, col], dtype=torch.long) for col in self.label_cols]

        # if self.transform:
        #     sample = self.transform(sample)

        return inputs, outputs
