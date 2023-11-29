import argparse
import numpy as np
import queue
from dataclasses import dataclass, field
from scipy.fft import fft
from audio_source import WaveFileAudioSource
from denoiser_preprocessing import DenoiserPreprocessing
from synthetic_signal_gen import SyntheticSignalGenerator
import plot_gen


@dataclass
class ExtractedFeatures:
    '''Data class to store extracted features'''
    pitch: float = 0
    band_energies: np.ndarray = field(default_factory=np.array)         # 24 band energies
    bfcc: np.ndarray = field(default_factory=np.array)                  # 24 BFCCs
    bfcc_first_derivs: np.ndarray = field(default_factory=np.array)     # First 6 BFCCs
    bfcc_second_derivs: np.ndarray = field(default_factory=np.array)    # First 6 BFCCs
    
class Denoiser():
    def __init__(self):
        self.denoise_preprocessor = DenoiserPreprocessing()
        
    def split_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return np.split(chunk, self.denoise_preprocessor.chunk_size / self.denoise_preprocessor.window_size)
    
    def extract_features(self, window):
        '''Extract the features from the window'''
        pitch = self.denoise_preprocessor.detect_pitch(window)
        
        windowed = self.denoise_preprocessor.apply_vorbis_window(window)
        windowed_fft = fft(windowed)
        band_energies = self.denoise_preprocessor.compute_band_energies(windowed_fft)
        bfcc = self.denoise_preprocessor.compute_bfcc(windowed_fft)
        derivs = self.denoise_preprocessor.compute_bcff_temporal_derivs(bfcc)
        
        print(f"Pitch: {pitch}, \
              Band Energies: {band_energies}, \
              BFCC: {bfcc}, \
              BFCC First Derivatives: {derivs[0]}, \
              BFCC Second Derivatives: {derivs[1]}")
        # return ExtractedFeatures(pitch, bfcc, band_energies)
        
    def analyze_windows(self, split_chunk: np.ndarray):
        features = []
        
        for i, window in enumerate(split_chunk):
            self.extract_features(window)
            
        return features
        
def main(input_path, output_path):
    denoiser = Denoiser()
    
    # Sine wave signal
    t = np.arange(0, 0.48, 1/denoiser.denoise_preprocessor.sample_rate)
    envelope = lambda t: np.exp(-t)
    sin_signal = envelope(t) * np.sin(2 * np.pi * 700 * t)
    
    chunk = denoiser.split_chunk(sin_signal)
    result = denoiser.analyze_windows(chunk)
    
    

    # sig_gen = SyntheticSignalGenerator()
    # path = sig_gen.synthetic_sin_signal(0.48, 100)
    # source = WaveFileAudioSource(path, DenoiserPreprocessing.SAMPLE_RATE)

    # audio_queue = queue.Queue()
    
    # source.read_into_queue(audio_queue)
    
    
    
    # counter = 0
    # while True:
    #     try:
    #         counter += 1
    #         chunk = audio_queue.get(block=False)
    #         chunk_int16 = np.copy(np.frombuffer(chunk, dtype=np.int16))
            
    #         if len(chunk) < DenoiserPreprocessing.CHUNK_SIZE:
    #             chunk_int16 = np.pad(chunk_int16, (0, DenoiserPreprocessing.CHUNK_SIZE - len(chunk_int16)))
            
    #         denoiser.analyze_window(chunk_int16)
            
    #     except queue.Empty:
    #         # Queue is empty, break out of the loop
    #         break
  
    
    
    # plot_gen.plot_freq_domain(X, DenoiserPreprocessing.SAMPLE_RATE, len(original), 'Original')
    # plot_gen.plot_freq_domain(X_denoised, DenoiserPreprocessing.SAMPLE_RATE, len(denoised), 'Denosied')
    
    # plot_gen.plot_time_domain(original, DenoiserPreprocessing.SAMPLE_RATE, 'Original')
    # plot_gen.plot_time_domain(denoised, DenoiserPreprocessing.SAMPLE_RATE, 'Denoised')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoiser")

    parser.add_argument('-i', '--input', required=True,
                        help="Path to noisy speech")
    parser.add_argument('-o', '--output', required=True,
                        help="Path to output denoised speech")
    
    args = parser.parse_args()
    
    main(args.input, args.output)