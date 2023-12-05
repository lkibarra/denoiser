import os
import numpy as np
import random
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from typing import Generator, Optional
from dataclasses import dataclass
from feature_extractor import FeatureExtractor

sample_rate = 16000
num_channels = 1

@dataclass
class SampleFileInfo:
    original_file_name: str
    original_file_path: Path
    noise_file_name: str
    noise_file_path: Path
    combined_file_path: Optional[Path] = None
    denoised_file_path: Optional[Path] = None

class DatasetGenerator(torch.utils.data.Dataset):
    def __init__(self, clean_sample_dir, noise_sample_dir, combined_sample_dir):
        self.clean_sample_dir = clean_sample_dir
        self.noise_sample_dir = noise_sample_dir
        self.combined_sample_dir = combined_sample_dir
        
        self.vad_model, self.utils = self.load_vad_model()
        
        self.dsp_feature_extractor = FeatureExtractor(sample_rate=16000, 
                                                      chunk_size=1536,
                                                      window_size=512, 
                                                      min_pitch=60, 
                                                      max_pitch=800)
        
        self.samples = self.generate_samples_info()
        
    def __len__(self):
        return len(list(self.samples))
    
    def __iter__(self):
        for sample in self.samples:
            # print(f"Getting audio sample: \n\
            #     Clean signal file name: {sample.original_file_name} \n\
            #     Clean signal file path: {sample.original_file_path} \n\
            #     Noise signal file name: {sample.noise_file_name} \n\
            #     Noise signal file path: {sample.noise_file_path} \n\
            #     Combined signal file path: {sample.combined_file_path}")
            
            clean_waveform, noise_waveform = self.format_audio(sample)
            
            expected_band_gains = self.compute_ideal_band_gains(clean_waveform, noise_waveform)

            combined_waveform = self.apply_noise(clean_waveform, noise_waveform)
            combined_waveform_windows = self.split_waveform(combined_waveform)
            
            vad_features = [self.get_vad_features(window) for window in combined_waveform_windows]
            dsp_features = [self.get_dsp_features(window) for window in combined_waveform_windows]
           
            dsp_features = torch.stack(dsp_features)
            vad_features = torch.stack(vad_features)    
            expected_band_gains = torch.from_numpy(expected_band_gains)
            
            yield dsp_features, vad_features, expected_band_gains

    def split_waveform(self, waveform):
        '''Split a waveform into 32ms windows'''
        
        windows = list(waveform.split(self.dsp_feature_extractor.window_size, dim=1))
        
        # Pad the last window if it is less than 32ms
        if windows[-1].shape[1] < self.dsp_feature_extractor.window_size:
            pad_size = self.dsp_feature_extractor.window_size - windows[-1].shape[1]
            windows[-1] = F.pad(windows[-1], (0, pad_size))
        
        return tuple(windows)
        
    def load_vad_model(self):
        '''Load the VAD model'''
        
        return torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                              model='silero_vad',
                              force_reload=True)
    
    def get_vad_features(self, waveform):
        '''Get the VAD features for a waveform'''
        
        with torch.no_grad():
            vad_features = self.vad_model(waveform, sample_rate)
            
        return vad_features
    
    def get_dsp_features(self, waveform):
        '''Get the DSP features for a waveform'''
        waveform = waveform.squeeze()
        
        return self.dsp_feature_extractor.extract_features(waveform)
    
    def compute_ideal_band_gains(self, clean_waveform, noisy_waveform):
        '''Compute the ideal per-band gains for a waveform'''
        clean_waveform_windows = self.split_waveform(clean_waveform)
        noisy_waveform_windows = self.split_waveform(noisy_waveform)
        
        gains = np.zeros((len(clean_waveform_windows), 24))
        for i, (clean_window, noisy_window) in enumerate(zip(clean_waveform_windows, noisy_waveform_windows)):
            clean_window = clean_window.squeeze()
            noisy_window = noisy_window.squeeze()
            
            band_indices = self.dsp_feature_extractor.bark_bands_indices()
            
            clean_fft = torch.fft.fft(clean_window).numpy()
            noisy_fft = torch.fft.fft(noisy_window).numpy()
            
            clean_bark_bands = self.dsp_feature_extractor.bark_bands(clean_fft, band_indices)
            noisy_bark_bands = self.dsp_feature_extractor.bark_bands(noisy_fft, band_indices)
            
            clean_band_energy = self.dsp_feature_extractor.compute_band_energies(clean_bark_bands)
            noisy_band_energy = self.dsp_feature_extractor.compute_band_energies(noisy_bark_bands)
            
            gains[i] = self.dsp_feature_extractor.calculate_per_band_gain(clean_band_energy, noisy_band_energy)
            
        return gains
            
        
    def apply_noise(self, clean_waveform, noise_waveform, mixing_ratio=0.4):
        '''Apply noise to a clean waveform'''
        
        if clean_waveform.size() != noise_waveform.size():
            raise ValueError("Waveforms must have the same length")
        
        combined_waveform = (1 - mixing_ratio) * clean_waveform + mixing_ratio * noise_waveform
        combined_waveform = F.normalize(combined_waveform, p=float('inf'), dim=1)
        
        return combined_waveform

    def generate_samples_info(self) -> Generator[SampleFileInfo, None, None]:
        '''Generate path info about the samples in the dataset'''
        
        for dir_path, _, filenames in os.walk(self.clean_sample_dir):
            for name in filenames:
            
            # for i, name in enumerate(filenames):
                # if i >= 100:  # For testing purposes, only generate one sample
                #     break
                
                if name.endswith(f".wav"):
                    new_dir_path = dir_path.replace(str(self.clean_sample_dir), str(self.combined_sample_dir))
                    combined_file_path = os.path.join(new_dir_path, name)
                    
                    noise_path = self.get_random_noise_file()
                    noise_name = os.path.basename(noise_path)
                    
                    sample_path = os.path.join(dir_path, name)
                    
                    yield SampleFileInfo(name, 
                                            Path(sample_path), 
                                            noise_name, 
                                            Path(noise_path), 
                                            Path(combined_file_path))
                        
    def get_random_noise_file(self) -> Path:
        files = [f for f in os.listdir(self.noise_sample_dir) if os.path.isfile(os.path.join(self.noise_sample_dir, f))]
        
        random_file_name = random.choice(files)
        
        return os.path.join(self.noise_sample_dir, random_file_name)
        
            
    def format_audio(self, sample: SampleFileInfo):
        clean_waveform, clean_sample_rate = torchaudio.load(sample.original_file_path)
        noise_waveform, noise_sample_rate = torchaudio.load(sample.noise_file_path)
        
        if clean_sample_rate != sample_rate:
            clean_waveform = torchaudio.transforms.Resample(clean_sample_rate, sample_rate)(clean_waveform)
        if noise_sample_rate != sample_rate:
            noise_waveform = torchaudio.transforms.Resample(noise_sample_rate, sample_rate)(noise_waveform)
            
        # Trim waveforms to the same length
        min_length = min(clean_waveform.shape[1], noise_waveform.shape[1])
        clean_waveform = clean_waveform[:, :min_length]
        noise_waveform = noise_waveform[:, :min_length]
        
        if clean_waveform.shape[0] != num_channels:
            clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)
        if noise_waveform.shape[0] != num_channels:
            noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)
        
        if clean_waveform.dtype != torch.int16:
            clean_waveform = F.normalize(clean_waveform, p=float('inf'), dim=1)
            clean_waveform = (clean_waveform * torch.iinfo(torch.int16).max).to(torch.int16)
        if noise_waveform.dtype != torch.int16:    
            noise_waveform = F.normalize(noise_waveform, p=float('inf'), dim=1)
            noise_waveform = (noise_waveform * torch.iinfo(torch.int16).max).to(torch.int16)
            
        return clean_waveform, noise_waveform