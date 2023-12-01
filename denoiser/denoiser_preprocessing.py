import numpy as np
from typing import Tuple
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
from functools import cached_property
from librosa import yin
from dataclasses import dataclass, field
from pprint import pformat

@dataclass
class ExtractedFeatures:
    '''Data class to store extracted features'''
    pitch: float = 0
    spectral_flux: float = 0
    bfcc: np.ndarray = field(default_factory=np.array)                  # 24 BFCCs
    bfcc_first_derivs: np.ndarray = field(default_factory=np.array)     # First 6 BFCCs
    bfcc_second_derivs: np.ndarray = field(default_factory=np.array)    # First 6 BFCCs
    pitch_correlation_dct: np.ndarray = field(default_factory=np.array) # First 6 pitch correlations
    
    def __str__(self):
        return f"Pitch: {self.pitch}\n\
                Spectral Flux: {self.spectral_flux}\n\
                BFCC: {pformat(self.bfcc)}\n\
                BFCC First Derivatives: {pformat(self.bfcc_first_derivs)}\n\
                BFCC Second Derivatives: {pformat(self.bfcc_second_derivs)}\n\
                Pitch Correlation DCT: {pformat(self.pitch_correlation_dct)}\n\
                \n Total Features: {self.count_features()}\n"
              
    def count_features(self):
        total_features = 0
        
        total_features += 2 # Pitch & Spectral Flux
        total_features += len(self.bfcc)
        total_features += len(self.bfcc_first_derivs)
        total_features += len(self.bfcc_second_derivs)
        total_features += len(self.pitch_correlation_dct)
        
        return total_features

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
        # 0 125 250 375 500 625 750 875 1k 1.125 1.25 1.5 1.75 2k 2.25 2.5 2.75 3.25k 3.75 4.25k 4.75 5.5 6.25k 7k 8k
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 26, 30, 34, 38, 44, 50, 56, 64
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
    
    def extract_features(self, window) -> ExtractedFeatures:
        '''Extract features from a window'''
        lpf_window = self.apply_butter_lowpass(window)
        
        pitch = self.detect_pitch(lpf_window)
        pitch_delayed_window = self.delay_window(lpf_window, pitch)
       
        windowed = self.apply_vorbis_window(lpf_window)
        delayed_windowed = self.apply_vorbis_window(pitch_delayed_window)
        
        windowed_fft = fft(windowed)
        delayed_windowed_fft = fft(delayed_windowed)
        
        band_indices = self.bark_bands_indices()
        bark_bands = self.bark_bands(windowed_fft, band_indices)
        
        flux = self.spectral_flux(windowed_fft)
        
        band_energies = self.compute_band_energies(bark_bands)
        
        bfcc = self.compute_bfcc(band_energies)
        temporal_derivatives = self.compute_bcff_temporal_derivs(bfcc)
        
        pitch_delayed_bark_bands = self.bark_bands(delayed_windowed_fft, band_indices)
        
        pitch_correlation = self.compute_pitch_correlation(bark_bands, pitch_delayed_bark_bands)
        
        pitch_correlation_dct = self.compute_pitch_dct(pitch_correlation)
        
        return ExtractedFeatures(pitch, 
                                 flux,
                                 bfcc, 
                                 temporal_derivatives[0][:6], 
                                 temporal_derivatives[1][:6],
                                 pitch_correlation_dct[:6]
                                 )
        
    
    def apply_vorbis_window(self, window) -> np.ndarray:
        '''Apply the Vorbis window with 0.5 window overlap'''
        
        vorbis = self.vorbis_coefficients
        
        for i in range(self.frame_size):
            window[i] *= vorbis[i]
            window[self.frame_size + i] *= vorbis[i] 
            
        return window
    
    def compute_bcff_temporal_derivs(self, bfcc) -> Tuple[np.ndarray, np.ndarray]:
        '''Compute the 1st and 2nd temporal derivatives of BFCC'''
        
        first_derivatives = np.gradient(bfcc)
        second_derivatives = np.gradient(first_derivatives)
        
        return first_derivatives, second_derivatives
    
    def calculate_per_band_gain(self, clean_band_energy, noisy_band_energy):
        gains = np.zeros(self.num_bands)
        
        for i in range(self.num_bands):
            gains[i] = np.sqrt(clean_band_energy[i] / noisy_band_energy[i])
            
        return gains
            
        
    def compute_band_energies(self, bands) -> np.ndarray:
        '''Calculate the energy in each band'''''
        
        band_energies = np.zeros(self.num_bands)
        total_amplitude = self.get_total_amplitude(bands)
        
        for i in range(self.num_bands - 1):
            band = bands[i]
            normalized_amplitude = np.sum(np.abs(band)) / total_amplitude
            
            power = (normalized_amplitude) * (np.abs(band) ** 2)
            
            band_size = len(band)
            triang_scale = np.arange(band_size) / band_size
            
            band_energies[i] += np.sum((1 - triang_scale) * power)
            band_energies[i + 1] += np.sum(triang_scale * power)
                    
        band_energies[0] *= 2
        band_energies[self.num_bands - 1] *= 2
        
        return band_energies

    def compute_bfcc(self, band_energies):
        '''Compute the Bark frequency cepstral coefficients'''
        
        epsilon = 1e-10     # To avoid log(0)
        log_bark_energies = np.log(band_energies + epsilon)
        dct_matrix = self.dct_matrix_coefficients
        
        bfcc = np.dot(dct_matrix, log_bark_energies)
        
        return bfcc
    
    def get_total_amplitude(self, bands):
        return np.sum([np.sum(np.abs(band)) for band in bands])
    
    def bark_bands_indices(self):
        '''Get the indices of the FFT corresponding to the Bark bands'''
        
        band_indices = []
        
        for i in range(self.num_bands):
            band_start = self.bark_band_scale[i] * 4 
            
            if i == self.num_bands - 1:
                band_end = self.bark_band_scale[-1] * 4
            else:
                band_end = self.bark_band_scale[i + 1] * 4
                
            band_indices.append((band_start, band_end))
            
        return band_indices
    
    def bark_bands(self, window_fft, band_indices):
        '''Split the FFT into bands according to the Bark scale'''
        
        bands = [window_fft[start : end] for start, end in band_indices]
        
        return bands
        
    def detect_pitch(self, window) -> float:
        '''Pitch detection based on the YIN algorithm'''
        pitches = yin(window, 
                      sr=self.sample_rate, 
                      fmin=self.min_pitch, 
                      fmax=self.max_pitch, 
                      frame_length=self.window_size)
        
        best_pitch_index = np.argmin(pitches)
        
        return pitches[best_pitch_index]
    
    def delay_window(self, window, pitch):
        '''Delay the window by the pitch'''
        
        pitch_period = int(self.sample_rate / pitch)
        
        return np.roll(window, pitch_period)
    
    def compute_pitch_dct(self, pitch_correlation):
        dct_matrix = self.dct_matrix_coefficients
        
        pitch_correlation_dct = np.dot(dct_matrix, pitch_correlation)
        return pitch_correlation_dct
    
    def compute_pitch_correlation(self, bands, pitch_delayed_bands):
        '''Compute the pitch correlation for each band'''
        
        pitch_correlation = np.zeros(self.num_bands)
        total_amplitude = self.get_total_amplitude(bands)
        
        for i in range (self.num_bands - 1):
            band = bands[i]
            normalized_amplitude = np.sum(np.abs(band)) / total_amplitude
            
            delayed_band = pitch_delayed_bands[i]
            
            numerator = np.sum(normalized_amplitude * np.real(band * np.conjugate(delayed_band)))
            denominator = np.sqrt(np.sum(normalized_amplitude * (np.abs(band) ** 2)) 
                                  * np.sum(normalized_amplitude * (np.abs(delayed_band) ** 2)))
            
            pitch_correlation[i] = numerator / denominator
            
        return pitch_correlation
    
    def spectral_flux(self, window_fft):
        '''Compute the spectral flux of the window'''
        
        flux = np.sum(np.square(np.diff(np.abs(window_fft))))
        return flux
    
    def apply_butter_lowpass(self, window):
        '''Apply a Butterworth lowpass filter to the window'''
        
        num, den = self.butter_lowpass()
        filtered_window_fft = lfilter(num, den, window)
        
        return filtered_window_fft
    
    def butter_lowpass(self, cutoff=8000, order=4):
        '''Butterworth lowpass filter'''
        
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        
        num, den = butter(order, normal_cutoff, btype='low', analog=False)
        
        return num, den
        