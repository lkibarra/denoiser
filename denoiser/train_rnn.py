import os
import torch
import torch.nn as nn
import denoiser_rnn 
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train(model, dataset_generator, vad_loss_fn, denoise_loss_fn, optimizer, num_epochs):
    model.train()
    
    writer = SummaryWriter("run")
    total_windows = 0
    print_interval = 200
    print('Starting training...')
    print('-----------------')
    
    for epoch in range(num_epochs):
        print(f'Epoch: [{epoch + 1}/{num_epochs}]')
        
        for i, (dsp_features, target_vad, target_band_gains) in enumerate(dataset_generator):
            optimizer.zero_grad()
            
            for window_idx in range(len(dsp_features)):
                model_input = dsp_features[window_idx]
                estimated_vad, estimated_band_gains = model(model_input)
                
                loss_vad_weight = 2 * torch.abs(target_vad[window_idx] - 0.5)
                loss_vad = torch.mean(loss_vad_weight * vad_loss_fn(estimated_vad, target_vad[window_idx]), dim=-1)
                
                # loss_denoise = (denoise_loss_fn(torch.sqrt(estimated_band_gains.squeeze()), torch.sqrt(target_band_gains[window_idx])))**0.5
                loss_denoise = denoise_loss_fn(torch.sqrt(estimated_band_gains.squeeze()), torch.sqrt(target_band_gains[window_idx]))
                loss_denoise = torch.mean(10 * torch.pow(loss_denoise, 4) + loss_denoise + 0.01 * vad_loss_fn(estimated_vad, target_vad[window_idx]), dim=-1)
                
                loss = 0.5 * loss_vad + 10 * loss_denoise  # Apply loss weights
                loss.requires_grad_(True)
                loss.backward()

                total_windows += 1
                
                if total_windows % print_interval == 0:
                    # print(f'Composite Loss: {loss.item()}')
                    # print(f'Estimated VAD: {estimated_vad.item()}, Target VAD: {target_vad[window_idx].item()}')
                    # print(f'Estimated Band Gains: {estimated_band_gains.mean().item()}, Target Band Gains: {target_band_gains[window_idx].mean().item()}')
                    # print('-----------------')
                    
                    writer.add_scalar('Composite Loss/train', loss.item(), total_windows)
                    writer.add_scalars('Loss', {'VAD loss': loss_vad.item(), 'Denoise loss': loss_denoise.item()}, total_windows)
                    writer.add_scalars('VAD/train', {'Estimated': estimated_vad.item(), 'Target': target_vad[window_idx].item()}, total_windows)
                    writer.add_scalars('Band Gains/train', {'Estimated': estimated_band_gains.mean().item(), 'Target': target_band_gains[window_idx].mean().item()}, total_windows)
                    writer.flush()
                    
            optimizer.step()
            # model.clip_weights()
            
            print(f'Percent complete: {round((i + 1) / len(dataset_generator) * 100, 2)}%')
        print('-----------------')
                
    print('Training finished.')
        
    
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


if __name__ == "__main__":
    project_path = os.getcwd()

    clean_dir = os.path.join(project_path, 'data', 'clean')
    noise_dir = os.path.join(project_path, 'data', 'noise', 'aircraft')
    combined_dir = os.path.join(project_path, 'data', 'combined')

    dataset_generator = DatasetGenerator(clean_dir, noise_dir, combined_dir)

    model = denoiser_rnn.DenoiserRNN(
            denoiser_rnn.input_size, 
            denoiser_rnn.output_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    vad_loss_fn = nn.BCELoss()
    denoise_loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=denoiser_rnn.learning_rate)

    train(model, dataset_generator, vad_loss_fn, denoise_loss_fn, optimizer, denoiser_rnn.num_epochs)

    torch.save(model.state_dict(), 'model.pth')

    print('Model saved and stored at model.pth')

    