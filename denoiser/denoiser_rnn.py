import torch
import torch.nn as nn

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# Hyperparameters
input_size = 44     # 44 input features
sequence_length = 1
hidden_size = 24
num_layers = 4
output_size = 24
num_epochs = 2
batch_size = 1
learning_rate = 0.01

band_mapping = {i: f"Bark Band Gains {i}" for i in range(24)}

class DenoiserRNN(nn.Module):
    '''Fully connected neural network with four hidden layer'''
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DenoiserRNN, self).__init__()
        
        gru_hidden_1 = hidden_size
        gru_hidden_2 = hidden_size * 2
        gru_hidden_3 = hidden_size * 4
        
        # input_data -> (batch_size, seq_length, feature_size)  What is the batch size and sequence length?
        
        self.dense_in = nn.Linear(input_size, gru_hidden_1) # Is the setup of the layers correct?
        
        self.gru_layers = [
            nn.GRU(gru_hidden_1, gru_hidden_1, batch_first=True),
            nn.GRU(gru_hidden_1, gru_hidden_2, batch_first=True),
            nn.GRU(gru_hidden_2, gru_hidden_3, batch_first=True)
        ]

        self.dense__out = nn.Linear(gru_hidden_3, output_size)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, input_data):
        input_data = input_data.float()
        out = self.tanh(self.dense_in(input_data))
        
        h0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(device)
        
        for i in range(len(self.gru_layers)):
            out, h0 = self.gru_layers[i](out, h0)
            out = self.relu(out)
        
        # output_data -> (batch_size, seq_length, hidden_size)
        
        out = out[:, -1, :] # How do we obtain the output from the last layer? Need the 22 band gains
        out = self.sigmoid(self.dense_out(out))
        
        return out
      
    def init_hidden(self):  # Is this needed?
        return torch.zeros(1, self.hidden_size) 