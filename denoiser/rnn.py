
import os
import torch
import torch.nn as nn
import torchaudio
from dataset_generator import DatasetGenerator

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# Hyperparameters
input_size = 44
sequence_length = 1
hidden_size = 24
num_layers = 4
output_size = 24
num_epochs = 2
batch_size = 1
learning_rate = 0.01

band_mapping = {i: f"Bark Band {i}" for i in range(24)}

class RNN(nn.Module):
    '''Fully connected neural network with four hidden layer'''
    
    def __init__(self):
        super(RNN, self).__init__()
        gru_hidden_1 = hidden_size
        gru_hidden_2 = hidden_size * 2
        gru_hidden_3 = hidden_size * 4
        
        # input_data -> (batch_size, seq_length, feature_size)
        
        self.dense_in = nn.Linear(input_size, gru_hidden_1)
        
        self.gru_layers = [
            nn.GRU(input_size, gru_hidden_1, batch_first=True),
            nn.GRU(gru_hidden_1, gru_hidden_2, batch_first=True),
            nn.GRU(gru_hidden_2, gru_hidden_3, batch_first=True)
        ]

        self.dense__out = nn.Linear(gru_hidden_3, output_size)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, input_data):
        out = self.tanh(self.dense_in(input_data))
        
        h0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(device)
        
        for i in range(len(self.gru_layers)):
            out, h0 = self.gru_layers[i](out, h0)
            out = self.relu(out)
        
        out = out[:, -1, :]
        out = self.sigmoid(self.dense_out(out))
        # output_data -> (batch_size, seq_length, hidden_size)
        
        return out
      
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    
def train_one_epoch(model, dataset_gen, loss_fn, optimizer):
    for i, (inputs, targets) in enumerate(dataset_gen.get):
        inputs, targets = inputs.to(device), targets.to(device) 
        
        predictions = model(inputs) # Outputs
        loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Loss: {loss.item()}')

def train(model, dataset_gen, loss_fn, optimizer, device, num_epochs):
    for i in range(num_epochs):
        print(f'Epoch {i + 1}')
        train_one_epoch(model, dataset_gen, loss_fn, optimizer, device)
        print('-----------------')
        
    print('Finished training.')
    
def predict(model, input, target, band_mapping):
    model.eval()
    with torch.no_grad():
        input, target = input.to(device), target.to(device)
        
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = band_mapping[predicted_index]
        expected = band_mapping[target.item()]
        
        print(f'Predicted: {predicted}')
        print(f'Target: {expected}')
    
    return predicted, expected

        
if __name__ == "__main__":
    
    project_path = os.getcwd()
    
    clean_dir = os.path.join(project_path, 'data', 'clean')
    noise_dir = os.path.join(project_path, 'data', 'noise', 'aircraft')
    aug_dir = os.path.join(project_path, 'data', 'augmented')
    
    dataset_gen = DatasetGenerator(clean_dir, noise_dir, aug_dir)
        
    model = RNN(input_size, hidden_size, num_layers, output_size)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, dataset_gen, loss_fn, optimizer, device, num_epochs)

    torch.save(model.state_dict(), 'model.pth')
    
    print('Model saved and stored at model.pth')
   
    
    
