import wave
import queue
import numpy as np
from typing import Tuple
from functools import cached_property

class WaveFileAudioSource():
    def __init__(self, file_path, sample_rate):
        self._file_path = file_path
        self._sample_rate = sample_rate
        self._ms_per_chunk = 96
        
    def read(self) -> Tuple[int, bytes]:
        '''Reads the audio file and returns the number of bytes read and the audio data'''
        
        with wave.open(self._file_path, 'rb') as audio_data:
            if audio_data.getsampwidth() != 2 or audio_data.getcomptype() != 'NONE':
                raise ValueError("Unsupported audio format. Only signed 16-bit little-endian PCM is supported")
            
            num_frames = audio_data.getnframes()
            print("audio_data.getframerate(): ")
            print(audio_data.getframerate())
            
            print("audio_data.getnchannels(): ")
            print(audio_data.getnchannels())
            
            print("audio_data.getsampwidth(): ")
            print(audio_data.getsampwidth())
            
            print("audio_data.getnframes(): ")
            print(audio_data.getnframes())
            
            if num_frames != self.sample_rate:
                raise ValueError("Unsupported audio sampling frequency. Only 16kHz is supported")
            
            data = audio_data.readframes(-1)
            bytes_read = num_frames * 2
            
            if bytes_read == 0:
                raise ValueError("No audio data found in file")
            
            return bytes_read, data
           