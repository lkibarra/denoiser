import argparse
import numpy as np
import queue
import librosa 
import padasip as pa
from audio_source import WaveFileAudioSource
from feature_extractor import FeatureExtractor

class Denoiser():
    def __init__(self, audio_source=None):
        self.audio_source = audio_source
        self.feature_extractor = FeatureExtractor(sample_rate=16000, 
                                                  chunk_size=1536, 
                                                  window_size=512, 
                                                  min_pitch=60, 
                                                  max_pitch=800)
        
    def split_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return np.split(chunk, self.feature_extractor.chunk_size / self.feature_extractor.window_size)
    
    def denoise_sub(self, chunk: np.ndarray, noise_chunk: np.ndarray, scaling_factor=1.5) -> np.ndarray:
        split_chunk = self.split_chunk(chunk)
        noise_split_chunk = self.split_chunk(noise_chunk)
        epsilon = 1e-10
        
        denoised_chunk = []
        
        for window, noise_window in (split_chunk, noise_split_chunk):
            window_stft = librosa.stft(window)
            noise_window_stft = librosa.stft(noise_window)
            
            window_mag = np.abs(window_stft)
            window_phase = np.angle(window_stft)
            
            noise_window_mag = np.abs(noise_window_stft)
            
            noise_subtracted = np.maximum(window_mag - scaling_factor * noise_window_mag, epsilon)
            
            denoised_window = librosa.istft(noise_subtracted * np.exp(1j * window_phase))
            denoised_chunk.append(denoised_window)
            
        return np.concatenate(denoised_chunk)
    
    def denoise_lms(self, chunk: np.ndarray) -> np.ndarray:
        split_chunk = self.split_chunk(chunk)
        filter = pa.filters.FilterLMS(n=22, mu=0.1, w="random")
        denoised_chunk = []
        
        for window in split_chunk:
            output, error, _ = filter.run(window, window)
            denoised_chunk.append(output)
            
        return np.concatenate(denoised_chunk)
        
    def denoise_rnn(self, chunk: np.ndarray) -> np.ndarray:
        split_chunk = self.split_chunk(chunk)
        
        denoised_chunk = []
        for window in split_chunk:
            features = self.analyze_windows(split_chunk)
            
            print(f'Window: {features}')
            
            # Placeholder: Add code to denoise with RNN model
            # Should call self.interpolate_band_gains() and self.apply_pitch_filter() to complete the denoising
            # Inverse FFT and window overlap-add should be done here to obtain the denoised window
            
        return np.concatenate(denoised_chunk)
        
        
def main(input_path, output_path, algorithm, plot):
    denoiser = Denoiser()
    
    if algorithm == 'sub':
        denoiser_func = denoiser.denoise_sub
    elif algorithm == 'lms':
        denoiser_func = denoiser.denoise_lms
    else:
        denoiser_func = denoiser.denoise_rnn
    
    source = WaveFileAudioSource(input_path)

    audio_queue = queue.Queue()
    
    source.read_into_queue(audio_queue)

    # Placeholder: Add code to write to output file
    original = []
    denoised = []
    
    while True:
        try:
            chunk = audio_queue.get(block=False)
            chunk_int16 = np.copy(np.frombuffer(chunk, dtype=np.int16))
            
            if len(chunk) < denoiser.feature_extractor.CHUNK_SIZE:
                chunk_int16 = np.pad(chunk_int16, (0, denoiser.feature_extractor.CHUNK_SIZE - len(chunk_int16)))
            
            original.append(chunk_int16)
            
            denoised_chunk = denoiser_func(chunk_int16)
            
            denoised.append(denoised_chunk)
            
        except queue.Empty:
            # Queue is empty, break out of the loop
            break
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoiser")

    parser.add_argument('-i', '--input', required=True,
                        help="Path to noisy speech")
    parser.add_argument('-o', '--output', required=True,
                        help="Path to output denoised speech")
    
    parser.add_argument('-a', '--algorithm', default='rnn', choices=['rnn', 'sub', 'lms'],
                        help="Denoising algorithm to use (default: rnn)")
    
    parser.add_argument('-p', '--plot', action='store_true', default=False, 
                        help="Plot the results")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.algorithm, args.plot)