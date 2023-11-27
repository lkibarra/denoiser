import argparse
import numpy as np
import queue
from audio_source import WaveFileAudioSource
from denoiser_preprocessing import DenoiserPreprocessing
from pitch import Pitch
from synthetic_signal_gen import SyntheticSignalGenerator
import plot_gen
from scipy.fft import fft, ifft, dct, idct

class Denoiser():
    def __init__(self):
        self.denoise_preprocessing = DenoiserPreprocessing()
        self.pitch_detector = Pitch(DenoiserPreprocessing.SAMPLE_RATE)
        
    def analyze_chunk(self, chunk: np.ndarray): # analyze 96ms of audio data (chunk size = 1536 bytes)
        windowed = self.denoise_preprocessing.apply_vorbis_window(chunk)
        
        pitch = self.pitch_detector.detect_pitch(chunk)
        print("pitch: ")
        print(pitch)
        
        
        return windowed
        # result = np.zeros(len(chunk), dtype=np.int16)
        
        # for i in range(0, len(chunk), self.denoise_preprocessing.WINDOW_SIZE):
        #     window = chunk[i : i + self.denoise_preprocessing.WINDOW_SIZE]
            
        #     # Apply Vorbis Window
        #     result[i : i + self.denoise_preprocessing.WINDOW_SIZE] = self.denoise_preprocessing.apply_vorbis_window(window.copy())
        # return result
            
        
def main(input_path, output_path):
    sig_gen = SyntheticSignalGenerator()
    path = sig_gen.synthetic_sin_signal(1, 100)
    
    source = WaveFileAudioSource(path,
                                DenoiserPreprocessing.SAMPLE_RATE)

    audio_queue = queue.Queue()
    
    source.read_into_queue(audio_queue)
    denoiser = Denoiser()
    
    original = []
    denoised = []
    
    counter = 0
    while True:
        try:
            counter += 1
            chunk = audio_queue.get(block=False)
            chunk_int16 = np.copy(np.frombuffer(chunk, dtype=np.int16))
            original.append(chunk_int16)
            
            windowed_chunk = denoiser.analyze_chunk(chunk_int16)
            denoised.append(windowed_chunk)
            
        except queue.Empty:
            # Queue is empty, break out of the loop
            break
  
    # print(f"The loop executed {counter} times.")
    original = np.concatenate(original)
    denoised = np.concatenate(denoised)
    
    X = fft(original)
    X_denoised = fft(denoised)
    
    plot_gen.plot_freq_domain(X, DenoiserPreprocessing.SAMPLE_RATE, len(original), 'Original')
    plot_gen.plot_freq_domain(X_denoised, DenoiserPreprocessing.SAMPLE_RATE, len(denoised), 'De')
    
    plot_gen.plot_time_domain(original, DenoiserPreprocessing.SAMPLE_RATE, 'Original')
    plot_gen.plot_time_domain(denoised, DenoiserPreprocessing.SAMPLE_RATE, 'Denoised')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoiser")

    parser.add_argument('-i', '--input', required=True,
                        help="Path to noisy speech")
    parser.add_argument('-o', '--output', required=True,
                        help="Path to output denoised speech")
    
    args = parser.parse_args()
    
    main(args.input, args.output)