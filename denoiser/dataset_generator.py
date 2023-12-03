import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from typing import Generator, Optional
from dataclasses import dataclass

sample_rate = 16000
num_channels = 1

@dataclass
class SampleFileInfo:
    original_file_name: str
    original_file_path: Path
    noise_file_name: str
    noise_file_path: Path
    augmented_file_path: Optional[Path] = None
    denoised_file_path: Optional[Path] = None

class DatasetGenerator(torch.utils.data.Dataset):
    def __init__(self, sample_dir, noise_dir, aug_dir):
        self.sample_dir = sample_dir
        self.noise_dir = noise_dir
        self.aug_dir = aug_dir
        
        self.samples = list(self.generate_samples_info())
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # print(f"Getting audio sample: \n\
            #     Clean file name: {sample.original_file_name} \n\
            #     Clean file path: {sample.original_file_path} \n\
            #     Noise file name: {sample.noise_file_name} \n\
            #     Noise file path: {sample.noise_file_path} \n\
            #     Augmented file path: {sample.augmented_file_path}")
        
        clean_waveform, noise_waveform = self.format_audio(sample)
        
        augmented_waveform = self.apply_noise(clean_waveform, noise_waveform)
        
        return clean_waveform, noise_waveform, augmented_waveform
    
 
    def apply_noise(self, clean_waveform, noise_waveform, mixing_ratio=0.4):
        '''Apply noise to a clean waveform'''
        
        if clean_waveform.size() != noise_waveform.size():
            raise ValueError("Waveforms must have the same length")
        
        combined_waveform = (1 - mixing_ratio) * clean_waveform + mixing_ratio * noise_waveform
        combined_waveform = F.normalize(combined_waveform, p=float('inf'), dim=1)
        
        return combined_waveform

    def generate_samples_info(self) -> Generator[SampleFileInfo, None, None]:
        '''Generate path info about the samples in the dataset'''
        
        for dir_path, _, filenames in os.walk(self.sample_dir):
            # for name in filenames:
            for i, name in enumerate(filenames):
                if i >= 1: 
                    break
                
                if name.endswith(f".wav"):
                    new_dir_path = dir_path.replace(str(self.sample_dir), str(self.aug_dir))
                    aug_file_path = os.path.join(new_dir_path, name)
                    
                    noise_path = self.get_random_noise_file()
                    noise_name = os.path.basename(noise_path)
                    
                    sample_path = os.path.join(dir_path, name)
                    
                    yield SampleFileInfo(name, 
                                            Path(sample_path), 
                                            noise_name, 
                                            Path(noise_path), 
                                            Path(aug_file_path))
                        
    def get_random_noise_file(self) -> Path:
        files = [f for f in os.listdir(self.noise_dir) if os.path.isfile(os.path.join(self.noise_dir, f))]
        
        random_file_name = random.choice(files)
        
        return os.path.join(self.noise_dir, random_file_name)
        
            
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