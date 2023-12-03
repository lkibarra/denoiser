import os
import torch
import torch.nn as nn
import denoiser_rnn 
from dataset_generator import DatasetGenerator
from denoiser import Denoiser

def train_one_epoch(model, dataset_generator, loss_fn, optimizer):
    for clean_waveform, noise_waveform, augmented_waveform in dataset_generator:
        model_input = torch.cat((clean_waveform, noise_waveform), dim=0)
        predictions = model(model_input)
        loss = loss_fn(predictions, augmented_waveform)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Loss: {loss.item()}')

def train(model, dataset_generator, loss_fn, optimizer, num_epochs):
    for i in range(num_epochs):
        print(f'Epoch {i + 1}')
        train_one_epoch(model, dataset_generator, loss_fn, optimizer)
        print('-----------------')
        
    print('Finished training.')
    
def predict(model, input, target, band_mapping):
    model.eval()
    with torch.no_grad():
        input, target = input.to(denoiser_rnn.device), target.to(denoiser_rnn.device)
        
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = band_mapping[predicted_index]
        expected = band_mapping[target.item()]
        
        print(f'Predicted: {predicted}')
        print(f'Target: {expected}')
    
    return predicted, expected

# TODO: Define own loss function
def loss_fn(gamma=0.5):
    # loss_fn = (estimated_band_gains ^ gamma) - (ideal_band_gains ^ gamma), where gamma = 0.5
    pass

        
if __name__ == "__main__":
    
    project_path = os.getcwd()
    
    clean_dir = os.path.join(project_path, 'data', 'clean')
    noise_dir = os.path.join(project_path, 'data', 'noise', 'aircraft')
    aug_dir = os.path.join(project_path, 'data', 'augmented')
    
    dataset_generator = DatasetGenerator(clean_dir, noise_dir, aug_dir)
    
    sample_file_info = next(dataset_generator)
    
    denoiser = Denoiser()
    
    model = denoiser_rnn.DenoiserRNN(
                denoiser_rnn.input_size, 
                denoiser_rnn.hidden_size, 
                denoiser_rnn.num_layers, 
                denoiser_rnn.output_size
    )
    
    loss_fn = nn.CrossEntropyLoss() # should define own loss function based on gain estimates and ground truth gains
    
    optimizer = torch.optim.Adam(model.parameters(), lr=denoiser_rnn.learning_rate)

    train(model, dataset_generator, loss_fn, optimizer, denoiser_rnn.num_epochs)

    torch.save(model.state_dict(), 'model.pth')
    
    print('Model saved and stored at model.pth')
   