import torch
import torch.nn as nn

# Hyperparameters
input_size = 44     # 44 input features
output_size = 24    # 24 bark band gains
num_layers = 4
num_epochs = 120
batch_size = 32
learning_rate = 0.01

band_mapping = {i: f"Bark Band Gains {i}" for i in range(24)}

class DenoiserRNN(nn.Module):
    '''Fully connected neural network with four hidden layer'''
    
    def __init__(self, input_size, output_size, hidden_size=24, num_layers=4):
        super(DenoiserRNN, self).__init__()

        self.dense_input_layer = nn.Linear(input_size, 24)
        self.gru_1_layer = nn.GRU(input_size=24, hidden_size=24, batch_first=True)
        self.gru_2_layer = nn.GRU(input_size=92, hidden_size=48, batch_first=True)
        self.gru_3_layer = nn.GRU(input_size=116, hidden_size=96, batch_first=True)
        self.dense_out_layer = nn.Linear(96, output_size)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, input_data):
        input_data = input_data.float()
        
        vad_dense_out = self.tanh(self.dense_input_layer(input_data))
        vad_dense_out.requires_grad_(True)
        
        # print(f'VAD Dense Out: {vad_dense_out.shape},\
        #     requires_grad: {vad_dense_out.requires_grad}\
        #     VAD Dense Out dtype: {vad_dense_out.dtype}')
        
        vad_gru_out_raw, _ = self.gru_1_layer(vad_dense_out)    #1x24
        vad_gru_out = self.relu(vad_gru_out_raw)
        vad_gru_out.requires_grad_(True)
        
        # print(f'VAD GRU Out: {vad_gru_out.shape},\
        #     requires_grad: {vad_gru_out.requires_grad},\
        #     VAD GRU Out dtype: {vad_gru_out.dtype}')
        
        noise_estimate_input = torch.cat((input_data, vad_dense_out, vad_gru_out), dim=1)
        noise_estimate_out_raw, _ = self.gru_2_layer(noise_estimate_input)    #1x92
        noise_estimate_out = self.relu(noise_estimate_out_raw)
        noise_estimate_out.requires_grad_(True)
        
        # print(f'Noise Estimate Out: {noise_estimate_out.shape},\
        #     requires_grad: {noise_estimate_out.requires_grad},\
        #     dtype: {noise_estimate_out.dtype}')
        
        band_gains_input = torch.cat((input_data, vad_gru_out, noise_estimate_out), dim=1)  #1x116
        band_gains_out_raw, _ = self.gru_3_layer(band_gains_input)
        band_gains_out = self.relu(band_gains_out_raw)
        band_gains_out.requires_grad_(True)
        
        # print(f'Band Gains Out: {band_gains_out.shape},\
        #     requires_grad: {band_gains_out.requires_grad},\
        #     dtype: {band_gains_out.dtype}')
        
        out = self.sigmoid(self.dense_out_layer(band_gains_out))
        out.requires_grad_(True)
        
        return out
    
    
    
    
        