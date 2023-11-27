import numpy as np
from dataclasses import dataclass, field
from functools import cached_property

@dataclass
class DenoiserPreprocessing:
    '''Data class to store speech preprocessing parameters'''
    
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1536 
    WINDOW_SIZE = 512
    FRAME_SIZE = int(WINDOW_SIZE / 2)    # Frame size in samples (half-window)
    WINDOW_OVERLAP_RATIO = 0.5

    NUM_BANDS = 24                  # Number of bands (approximation of the Bark scale)
    BARK_BAND_SCALE = [
    # 0 125 250 375 500 750 875 1k 1.125 1.25 1.375 1.5 1.75 2k 2.25 2.5 3k 3.5 4k 4.5 5.25 6k 7k 8k
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 22, 26, 30, 34, 40, 46, 54, 62, 100
    ]
    
    @cached_property
    def vorbis_coefficients(self) -> np.ndarray:
        i = np.arange(self.FRAME_SIZE, dtype=float)
        vorbis_window_coefficients = np.sin((0.5 * np.pi) * (np.sin(0.5 * np.pi * (i + 0.5) / self.FRAME_SIZE) ** 2))
        
        return vorbis_window_coefficients
    
        # print("Vorbis Window Coefficients:", len(self.vorbis_window_coefficients))
        # print(np.round(self.vorbis_window_coefficients, 3))
    
    @cached_property
    def dct_matrix_coefficients(self) -> np.ndarray:
        self.dct_coefficients_matrix = np.zeros((self.NUM_BANDS, self.NUM_BANDS), dtype=float)
        
        for i in range(self.NUM_BANDS):
            for j in range(self.NUM_BANDS):
                if i == 0:
                    self.dct_coefficients_matrix[i, j] = np.sqrt(1 / self.NUM_BANDS)
                else :
                    self.dct_coefficients_matrix[i, j] = \
                        np.sqrt(2 / self.NUM_BANDS) * np.cos((np.pi / self.NUM_BANDS) * i * (j + 0.5))
                        
        return self.dct_coefficients_matrix       
        # print("DCT Coefficients Matrix:")
        # print(np.round(self.dct_coefficients_matrix, 3)) 
        
    def apply_vorbis_window(self, window):
        vorbis = self.vorbis_coefficients
        
        for i in range(len(vorbis)):
            window[i] *= vorbis[i]
            window[-i - 1] *= vorbis[i] 
            
        return window
    
    def apply_dct(self, x):
        y = np.dot(self.dct_coefficients_matrix.T, x)
        return y
    
    def bark_bands(self, X):    # X is the FFT of the windowed signal 
        '''Split the FFT into bands according to the Bark scale'''
        
        bands = []
        total_amplitude = 0
        for i in range(self.NUM_BANDS - 1):
            band_start = self.BARK_BAND_SCALE[i] * 4 
            band_end = self.BARK_BAND_SCALE[i + 1] * 4
            band = X[band_start : band_end]
            
            bands.append(band)
            total_amplitude += np.sum(np.abs(band))
        
        print("num_bands: ")
        print(len(bands))
        
        print("total_amplitude: ")
        print(total_amplitude)
        
        return bands, total_amplitude
    
    def normalize_amplitudes(self, bands, total_amplitude):
        normalized_bands = [band / total_amplitude for band in bands]
        return normalized_bands
        
    def compute_band_energies(self, X):
        '''Calculate the energy in each band'''''
        band_energies = []
        bands, total_amplitude = self.bark_bands(X)
        normalized_bands = self.normalize_amplitudes(bands, total_amplitude)
        for i in range(self.NUM_BANDS - 1):
            band = normalized_bands[i]
            power = np.abs(band) ** 2
            
            band_size = len(band)
            triang_scale = np.arange(band_size) / band_size
            
            band_energies[i] += np.sum((1 - triang_scale) * power)
            band_energies[i + 1] += np.sum(triang_scale * power)
                    
        band_energies[0] *= 2
        band_energies[self.NUM_BANDS - 1] *= 2
        
        return band_energies
    
    