import torch
import torch.nn as nn

# Hyperparameters
input_size = 44     # 44 input features
output_size = 24    # 24 bark band gains
num_layers = 4
num_epochs = 10
learning_rate = 0.01

band_mapping = {i: f"Bark Band Gains {i}" for i in range(24)}

class DenoiserRNN(nn.Module):
    '''Fully connected neural network with four hidden layer'''
    
    def __init__(self, input_size, output_size, hidden_size=24, num_layers=4):
        super(DenoiserRNN, self).__init__()

        self.dense_input_layer = nn.Linear(input_size, 24)
        self.vad_gru = nn.GRU(input_size=24, hidden_size=24, batch_first=True)
        self.vad_output = nn.Linear(24, 1)  
        self.noise_gru = nn.GRU(input_size=92, hidden_size=48, batch_first=True)
        self.denoise_gru = nn.GRU(input_size=116, hidden_size=96, batch_first=True)
        self.denoise_output = nn.Linear(96, output_size)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, input_data):
        input_data = input_data.float()
        
        vad_dense_out = self.tanh(self.dense_input_layer(input_data))
        vad_dense_out.requires_grad_(True)
        
        vad_gru_out_raw, _ = self.vad_gru(vad_dense_out)    #1x24
        # vad_gru_out = self.relu(vad_gru_out_raw)
        vad_gru_out = self.tanh(vad_gru_out_raw)
        vad_gru_out.requires_grad_(True)
        
        vad_out = self.sigmoid(self.vad_output(vad_gru_out))
        vad_out.requires_grad_(True)
        
        noise_input = torch.cat((input_data, vad_dense_out, vad_gru_out), dim=1)
        
        noise_gru_out_raw, _ = self.noise_gru(noise_input)  #1x92
        noise_gru_out = self.relu(noise_gru_out_raw)
        noise_gru_out.requires_grad_(True)
        
        denoise_input = torch.cat((input_data, vad_gru_out, noise_gru_out), dim=1)  #1x116
        
        denoise_gru_out_raw, _ = self.denoise_gru(denoise_input)
        denoise_gru_out = self.tanh(denoise_gru_out_raw)
        denoise_gru_out.requires_grad_(True)
        
        denoise_out = self.sigmoid(self.denoise_output(denoise_gru_out))
        denoise_out.requires_grad_(True)
        
        # print("--------------------------------------------------")
        # print(f'VAD Out: {vad_out.shape},\
        #     requires_grad: {vad_out.requires_grad},\
        #     dtype: {vad_out.dtype}')
        
        # print(f'Denoise Out: {denoise_out.shape},\
        #     requires_grad: {denoise_out.requires_grad},\
        #     dtype: {denoise_out.dtype}')
        # print("--------------------------------------------------")
        
        return vad_out, denoise_out
    
    # def clip_weights(self, min_value=-0.499, max_value=0.499):
    #     for param in self.parameters():
    #         param.data.clamp_(min_value, max_value)
    
    
    
    
        