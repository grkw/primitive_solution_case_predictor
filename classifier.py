import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a simple model
class FCNStateSelector(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCNStateSelector, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(64, 128)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout2(x)
        # x = self.fc3(x)
        # x = F.relu(self.bn3(x))
        # x = self.dropout3(x)
        x = self.fc4(x)
        return x
    
# Could conv along each x,y,z axis, along each waypoint
class CNNStateSelector(nn.Module):
    def __init__(self):
        super(CNNStateSelector, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, 5)
        self.conv2 = nn.Conv1d(20, 50, 5)
        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, (18*2+1)*(19*2+1))
        # self.fc2 = nn.Linear(10, 1) #output is a continuous value but since the outputs are discrete, we need to use a softmax layer
        # should train like a classification problem since I have that for the outputs and anything more granular wouldn't make a signficant difference in execution time, and to handle -1,0 case
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 50)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.Softmax(x)
        return x

# Input sequence may be so short that LSTM is not necessary
class LSTMStateSelector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMStateSelector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out