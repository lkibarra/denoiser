import numpy as np
from dataclasses import dataclass, field

@dataclass
class DenoiserPreprocessing:
    '''Data class to store speech preprocessing parameters'''
    
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 512               # Window size in samples
    FRAME_SIZE = int(WINDOW_SIZE / 2)    # Frame size in samples (half-window)
    WINDOW_OVERLAP_RATIO = 0.5

    NUM_BANDS = 24                  # Number of bands (approximation of the Bark scale)
    BARK_BAND_SCALE = [
    # 0 125 250 375 500 750 875 1k 1.125 1.25 1.375 1.5 1.75 2k 2.25 2.5 3k 3.5 4k 4.5 5.25 6k 7k 8k
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 22, 26, 30, 34, 40, 46, 54, 62, 100
    ]
    
    vorbis_window_coefficients: np.ndarray = field(init=False, default=None) 
    dct_coefficients_matrix: np.ndarray = field(init=False, default=None)
    
    def __init__(self):
        self.vorbis_window_coefficients = self.compute_vorbis_window_coeff()
        self.dct_coefficients_matrix = self.compute_dct_coeff()