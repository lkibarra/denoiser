import numpy as np
import time
import pyaudio
import wave
import os

class SyntheticSignalGenerator():
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.volume = 1
        
    def play_audio(self, audio_data: bytes):
        # for paInt16 sample values must be in range [-32768, 32767]
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        output=True)
        
        # play. May repeat with different volume values (if done interactively)
        start_time = time.time()
        stream.write(audio_data)
        print("Played sound for {:.2f} seconds".format(time.time() - start_time))

        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def write_audio(self, audio_data: bytes) -> str:
        curr_dir = os.getcwd()
        file = "tests/synthetic_sig.wav"
        filepath = os.path.join(curr_dir, file)
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
        with wave.open(filepath, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 2 bytes because np.int16 is used
            f.setframerate(self.sample_rate)
            f.writeframes(audio_data)

        return filepath
        
    def synthetic_sin_signal(self, duration_s, freq_hz=100) -> str:
        t = np.arange(0, duration_s, 1 / self.sample_rate)
        
        sin_signal = self.volume * np.sin(2 * np.pi * freq_hz * t)
        scaled = np.int16(sin_signal * 32767)
        data = scaled.tobytes()
        
        return self.write_audio(data)
