import os
import torch
import torch.nn as nn
import denoiser_rnn 
from dataset_generator import DatasetGenerator


def train(model, dataset_generator, loss_fn, optimizer, num_epochs):
    
    print('Starting Model training')
    print('-----------------')
    total_batches = num_epochs * denoiser_rnn.batch_size
    
    batch_count = 0
    for epoch in range(num_epochs):
        print(f'Epoch: [{epoch + 1}/{num_epochs}]')
        
        for i, (dsp_features, vad_values, expected_band_gains) in enumerate(dataset_generator):
            print(f'Iteration: {i + 1}')
            
            optimizer.zero_grad()
            
            model_input = dsp_features
            estimated = model(model_input)
            
            loss = loss_fn(estimated, expected_band_gains, vad_values)
           
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            
            print(f'Loss: {loss.item()}')
            
            batch_count += 1
            if batch_count >= total_batches:
                break
            
            if batch_count % denoiser_rnn.batch_size == 0:
                print('-----------------')
                print(f'Starting new epoch after {denoiser_rnn.batch_size} batches')
                break
            
    print('Finished training.')
        
    
# def predict(model, input, target, band_mapping):
#     model.eval()
#     with torch.no_grad():
#         input, target = input.to(denoiser_rnn.device), target.to(denoiser_rnn.device)
        
#         predictions = model(input)
#         predicted_index = predictions[0].argmax(0)
#         predicted = band_mapping[predicted_index]
#         expected = band_mapping[target.item()]
        
#         print(f'Predicted: {predicted}')
#         print(f'Target: {expected}')
    
#     return predicted, expected

def loss_fn(estimated, expected, vad_values):
    
    mse_loss = nn.MSELoss()
    mse = mse_loss(expected, estimated)
    loss = mse ** 0.25
    
    # ce_loss = nn.CrossEntropyLoss()
    # cross_entropy = ce_loss(estimated.long(), vad_values.long())
    
    return loss

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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=denoiser_rnn.learning_rate)

    train(model, dataset_generator, loss_fn, optimizer, denoiser_rnn.num_epochs)

    torch.save(model.state_dict(), 'model.pth')

    print('Model saved and stored at model.pth')

    