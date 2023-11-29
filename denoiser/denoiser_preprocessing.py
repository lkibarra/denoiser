import numpy as np
from typing import Tuple
from functools import cached_property
from librosa import feature, pyin


class DenoiserPreprocessing:
    '''Preprocessing class for the denoiser'''
    
    def __init__(self, sample_rate=16000, chunk_size=1536, window_size=512, min_pitch=60, max_pitch=800):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.frame_size = int(window_size / 2)
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        
        self.num_bands = 24
        self.bark_band_scale = [
        # 0 125 250 375 500 750 875 1k 1.125 1.25 1.375 1.5 1.75 2k 2.25 2.5 3k 3.5 4k 4.5 5.25 6k 7k 8k
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 22, 26, 30, 34, 40, 46, 54, 62, 100
        ]
        
    @cached_property
    def vorbis_coefficients(self) -> np.ndarray:
        i = np.arange(self.frame_size, dtype=float)
        vorbis_window_coefficients = np.sin((0.5 * np.pi) * (np.sin(0.5 * np.pi * (i + 0.5) / self.frame_size) ** 2))
        
        return vorbis_window_coefficients

    @cached_property
    def dct_matrix_coefficients(self) -> np.ndarray:
        self.dct_coefficients_matrix = np.zeros((self.num_bands, self.num_bands), dtype=float)
        
        for i in range(self.num_bands):
            for j in range(self.num_bands):
                if i == 0:
                    self.dct_coefficients_matrix[i, j] = np.sqrt(1 / self.num_bands)
                else :
                    self.dct_coefficients_matrix[i, j] = \
                        np.sqrt(2 / self.num_bands) * np.cos((np.pi / self.num_bands) * i * (j + 0.5))
                        
        return self.dct_coefficients_matrix       
    
    def apply_vorbis_window(self, window) -> np.ndarray:
        '''Apply the Vorbis window with 0.5 window overlap'''
        
        vorbis = self.vorbis_coefficients
        
        for i in range(self.frame_size):
            window[i] *= vorbis[i]
            window[self.frame_size + i] *= vorbis[i] 
            
        return window
    
    def apply_dct(self, window) -> np.ndarray:
        return np.dot(self.dct_coefficients_matrix.T, window)
    
    def compute_bcff_temporal_derivs(self, bfcc, num=6) -> Tuple[np.ndarray, np.ndarray]:
        '''Compute the 1st and 2nd temporal derivatives of first 6 BFCCs'''
        selected = bfcc[:num]
        
        first_derivatives = [np.gradient(bfcc) for bfcc in selected]
        second_derivatives = [np.gradient(first_deriv) for first_deriv in first_derivatives]
        return first_derivatives, second_derivatives
        
    def compute_band_energies(self, window_fft) -> np.ndarray:
        '''Calculate the energy in each band'''''
        
        bark_bands, amplitude = self.bark_bands(window_fft)
        normalized_bands = self.normalize_amplitude(bark_bands, amplitude)
        band_energies = np.zeros(self.num_bands)
        
        for i in range(self.num_bands - 1):
            band = normalized_bands[i]
            power = np.abs(band) ** 2
            
            band_size = len(band)
            triang_scale = np.arange(band_size) / band_size
            
            band_energies[i] += np.sum((1 - triang_scale) * power)
            band_energies[i + 1] += np.sum(triang_scale * power)
                    
        band_energies[0] *= 2
        band_energies[self.num_bands - 1] *= 2
        
        return band_energies
    
    def compute_bfcc(self, window_fft):
        '''Compute the Bark frequency cepstral coefficients'''
        bark_bands, amplitudes = self.bark_bands(window_fft)
        epsilon = 1e-10     # To avoid log(0)
        
        return [np.log(np.abs(band) + epsilon) for band in bark_bands]
    
    def bark_bands(self, window_fft):
        '''Split the FFT into bands according to the Bark scale'''
        
        bands = []
        total_amplitude = 0
        for i in range(self.num_bands - 1):
            band_start = self.bark_band_scale[i] * 4 
            band_end = self.bark_band_scale[i + 1] * 4
            band = list(window_fft[band_start : band_end])
            
            bands.append(band)
            total_amplitude += np.sum(np.abs(band))
            
        return bands, total_amplitude
    
    def normalize_amplitude(self, bands, total_amplitude):
        '''Normalize the band amplitudes'''
        
        return [band / total_amplitude for band in bands]
    
    def detect_pitch(self, window, threshold = 0.1) -> float:
        '''Pitch detection based on the YIN algorithm'''
        pitches, voiced_flag, voiced_probs = pyin(window, 
                                                  sr=self.sample_rate, 
                                                  fmin=self.min_pitch, 
                                                  fmax=self.max_pitch)
        best_pitch_index = np.argmax(voiced_probs)
        
        return pitches[best_pitch_index]
    