import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv

class ConvertCSVToExamples():
    def __init__(self, csv_fname):
        self.data = pd.read_csv(csv_fname)
        print("Original data shape: ", self.data.shape)
        # Delete rows with invalid input
        self.data = self.data[self.data['ruckig_solution_x'] != 'INVALID_INPUT']
        self.data = self.data.reset_index(drop=True)

        # Replace _UDDU or _UDUD with empty string
        for col in ['ruckig_solution_x','ruckig_solution_y','ruckig_solution_z']:
            self.data[col] = self.data[col].replace('_UDUD|_UDDU', '', regex=True) 

        self.indep_min_x_loc = int(self.data.columns.get_loc('indep_min_x'))

        print("Data shape after removing invalids: ", self.data.shape)

    def generateExamplesCSV(self, output_fname):

        solution_function_list = ['ACC0_ACC1_VEL', 'ACC1_VEL', 'ACC0_VEL', 'VEL', 'ACC0_ACC1', 'ACC1', 'ACC0', 'NONE']

        print("Generating examples")
        num_rows = self.data.shape[0] * 2  # Each row in self.data should generate exactly 2 rows in the new DataFrame
        dataset = pd.DataFrame(index=range(num_rows), columns=['pi', 'vi', 'ai', 'pf', 'vf', 'af', 't_profile', 'ruckig_solution', 'label'])

        row_idx = 0
        for idx in range(self.data.shape[0]):  # Iterate over all rows
            limiting_dof = int(self.data.at[idx, 'limiting_dof'])
            t_profile = self.data.iat[idx, self.indep_min_x_loc+limiting_dof]

            for dof_idx in range(3):  # 0, 1, 2
                if (limiting_dof == dof_idx):
                    continue

                dataset.at[row_idx, 'pi'] = self.data.iat[idx, 0+dof_idx]
                dataset.at[row_idx, 'vi'] = self.data.iat[idx, 3+dof_idx]
                dataset.at[row_idx, 'ai'] = self.data.iat[idx, 6+dof_idx]
                dataset.at[row_idx, 'pf'] = self.data.iat[idx, 9+dof_idx]
                dataset.at[row_idx, 'vf'] = self.data.iat[idx, 12+dof_idx]
                dataset.at[row_idx, 'af'] = self.data.iat[idx, 15+dof_idx]
                dataset.at[row_idx, 't_profile'] = t_profile
                ruckig_solution = self.data.iat[idx, 18+dof_idx]
                dataset.at[row_idx, 'ruckig_solution'] = ruckig_solution
                
                direction = ruckig_solution.split("_", 1)[0]
                solution_name = ruckig_solution.split("_", 1)[1]
                solution_id = solution_function_list.index(solution_name)
                if direction == 'DOWN':
                    solution_id += 8
                dataset.at[row_idx, 'label'] = solution_id
                
                row_idx += 1
        
        dataset = dataset.dropna()  # Remove any unused rows

        dataset.to_csv(output_fname, index=False)

        labels = dataset['label'].tolist()
        with open('solution_labels.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the list to the file
            writer.writerow(labels)

# AFTER GENERATING EXAMPLES, USE THE EXAMPLES CSV TO CREATE A DATASET
class CSVDataset(Dataset):
    def __init__(self, csv_files, input_col, label_col, num_sample):

        print("Loading data from CSV files")
        print("CSV files: ", csv_files)
        
        # Merge the data from all the CSV files into a single DataFrame. Automatically treats the first row as the header and does not include it in the data
        self.data = pd.concat([pd.read_csv(f) for f in csv_files])

        print("Data shape: ", self.data.shape)

        print("\nData distributions by class")
        self.le = LabelEncoder()

        self.data[label_col] = self.le.fit_transform(self.data[label_col])
        print(f"Class labels shape: ", self.le.classes_.shape)

        label_distribution = (np.bincount(self.data[label_col])/self.data.shape[0]*100).round(2)
        label_raw_numbers = np.bincount(self.data[label_col])
        # Print the classes and their frequencies
        class_distribution = list(zip(self.le.classes_, label_raw_numbers, label_distribution))
        
        class_distribution_df = pd.DataFrame(class_distribution, columns=['Class', 'Number of examples', 'Frequency (%)'])
        class_distribution_df = class_distribution_df.sort_values(by='Frequency (%)', ascending=False)
        print(class_distribution_df)
        class_distribution_df.to_csv(f'dataset_distr.csv', index=False)

        self.label_cols = label_col
        self.input_col = input_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.data.iloc[idx, :self.input_col]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        
        outputs = [torch.tensor(self.data.iloc[idx, col], dtype=torch.long) for col in self.label_cols]

        return inputs, outputs

if __name__ == "__main__":
    csv_fname = 'data/data_1scale_5seed_5vmax_5amax_15jmax_raw.csv'
    examples = ConvertCSVToExamples(csv_fname)
    examples.generateExamplesCSV('data/data_1scale_5seed_5vmax_5amax_15jmax.csv')
    # train_dataset = CSVDataset(cfg.train_csv_file, cfg.csv_input_col, cfg.csv_label_col, -999)