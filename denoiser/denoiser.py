import argparse
import numpy as np
import queue
import librosa 
from audio_source import WaveFileAudioSource
from feature_extractor import FeatureExtractor

class Denoiser():
    def __init__(self):
        self.feature_extractor = FeatureExtractor(sample_rate=16000, 
                                                  chunk_size=1536, 
                                                  window_size=512, 
                                                  min_pitch=60, 
                                                  max_pitch=800)
        
    def split_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return np.split(chunk, self.feature_extractor.chunk_size / self.feature_extractor.window_size)
    
    def denoise_lms(self, chunk: np.ndarray) -> np.ndarray:
        split_chunk = self.split_chunk(chunk)
        
        denoised_chunk = []
        
        for window in split_chunk:
            window_stft = librosa.stft(window)
            
            estimated_noise = np.median(window_stft, axis=1)
            
            noise_subtracted = np.abs(window_stft - estimated_noise[:, np.newaxis])
            
            denoised_window = librosa.istft(noise_subtracted)
            denoised_chunk.append(denoised_window)
            
        return np.concatenate(denoised_chunk)
        
    def denoise_rnn(self, chunk: np.ndarray) -> np.ndarray:
        split_chunk = self.split_chunk(chunk)
        
        denoised_chunk = []
        for window in split_chunk:
            features = self.analyze_windows(split_chunk)
            
            print(f'Window: {features}')
            
            # Do the RNN stuff here
            
        return np.concatenate(denoised_chunk)
    
    # def iterpolate_band_gains(self, band_gains):
    
    # def apply_pitch_filter(self, band_filter_coeff):
        
def main(input_path, output_path, algorithm, plot):
    # Sine wave signal
    # t = np.arange(0, 0.48, 1/denoiser.denoise_preprocessor.sample_rate)
    # envelope = lambda t: np.exp(-t)
    # sin_signal = envelope(t) * np.sin(2 * np.pi * 100 * t)
    
    # num_chunks = int(len(sin_signal) / denoiser.denoise_preprocessor.chunk_size)
    # sin_signal_chunks = np.split(sin_signal, num_chunks)    # Audio queue will already contain chunks
    
    # for chunk in sin_signal_chunks:
    #     split_chunk = denoiser.split_chunk(chunk)
    #     denoiser.analyze_windows(split_chunk)
    
    denoiser = Denoiser()
    
    if algorithm == 'lms':
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
    
    parser.add_argument('-a', '--algorithm', default='rnn', choices=['rnn', 'lms'],
                        help="Denoising algorithm to use (default: rnn)")
    
    parser.add_argument('-p', '--plot', action='store_true', default=False, 
                        help="Plot the results")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.algorithm, args.plot)