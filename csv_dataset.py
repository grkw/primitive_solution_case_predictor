import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class ConvertCSVToExamples():
    def __init__(self, csv_fname):
        self.data = pd.read_csv(csv_fname)
        print("Original data shape: ", self.data.shape)
        # Delete rows with invalid input
        self.data = self.data[self.data['ruckig_solution_x'] != 'INVALID_INPUT']
        self.data = self.data.reset_index(drop=True)

        # Replace _UDDU or _UDUD with empty string
        self.data['ruckig_solution_x'] = self.data['ruckig_solution_x'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_y'] = self.data['ruckig_solution_y'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_z'] = self.data['ruckig_solution_z'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_x'] = self.data['ruckig_solution_x'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_y'] = self.data['ruckig_solution_y'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_z'] = self.data['ruckig_solution_z'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_x'] = self.data['ruckig_solution_x'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_y'] = self.data['ruckig_solution_y'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_z'] = self.data['ruckig_solution_z'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_x'] = self.data['ruckig_solution_x'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_y'] = self.data['ruckig_solution_y'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_z'] = self.data['ruckig_solution_z'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_x'] = self.data['ruckig_solution_x'].replace('_UDUD|_UDDU', '', regex=True)
        self.data['ruckig_solution_y'] = self.data['ruckig_solution_y'].replace('_UDUD|_UDDU', '', regex=True)

        self.indep_min_x_loc = int(self.data.columns.get_loc('indep_min_x'))

        print("Data shape after removing invalids: ", self.data.shape)

    def generateExamplesCSV(self, output_fname):

        dataset = pd.DataFrame(columns=['pi', 'vi', 'ai', 'pf', 'vf', 'af', 't_profile', 'ruckig_solution'])
        print("Generating examples")
        for idx in range(self.data.shape[0]): # Iterate over all rows
            print(idx)
            limiting_dof = int(self.data.at[idx, 'limiting_dof'])
            t_profile = self.data.iat[idx, self.indep_min_x_loc+limiting_dof]
            
            for dof_idx in range(3): # 0, 1, 2
                
                if (limiting_dof == dof_idx):
                    continue

                pi = self.data.iat[idx, 0+dof_idx]
                vi = self.data.iat[idx, 3+dof_idx]
                ai = self.data.iat[idx, 6+dof_idx]
                pf = self.data.iat[idx, 9+dof_idx]
                vf = self.data.iat[idx, 12+dof_idx]
                af = self.data.iat[idx, 15+dof_idx]
                ruckig_solution = self.data.iat[idx, 18+dof_idx]
                
                new_row = pd.DataFrame({'pi': [pi], 'vi': [vi], 'ai': [ai], 'pf': [pf], 'vf': [vf], 'af': [af], 't_profile': [t_profile], 'ruckig_solution': [ruckig_solution]})
                dataset = pd.concat([dataset, new_row], ignore_index=True)
        dataset.to_csv(output_fname, index=False)

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

if __name__ == "__main__":
    csv_fname = 'data/train_1scale_7seed_5max_5amax.csv'
    examples = ConvertCSVToExamples(csv_fname)
    examples.generateExamplesCSV('data/train_1scale_7seed_5max_5amax_examples.csv')