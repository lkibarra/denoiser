import wave
import queue
import numpy as np
from typing import Tuple
from functools import cached_property

class WaveFileAudioSource():
    def __init__(self, file_path):
        self._file_path = file_path
        self._sample_rate = 16000
        self._ms_per_chunk = 96
        
    def read(self) -> Tuple[int, bytes]:
        '''Reads the audio file and returns the number of bytes read and the audio data'''
        
        with wave.open(self._file_path, 'r') as audio_data:
            if audio_data.getsampwidth() != 2 or audio_data.getcomptype() != 'NONE':
                raise ValueError("Unsupported audio format. Only signed 16-bit little-endian PCM is supported")
        
            if audio_data.getframerate() != self.sample_rate:
                raise ValueError("Unsupported audio sampling frequency. Only 16kHz is supported")
            
            data = audio_data.readframes(-1)
            bytes_read = audio_data.getnframes() * 2
            
            if bytes_read == 0:
                raise ValueError("No audio data found in file")
            
            return bytes_read, data
           
    def read_into_queue(self, data_queue: queue.Queue) -> int:
        '''Read audio chunks and place the data into the queue'''
        
        bytes_read, data = self.read()
        counter = 0
        if bytes_read > 0 and data_queue is not None:
            for i in range(0, len(data), self.bytes_per_chunk):
                counter += 1
                chunk = data[i:i+self.bytes_per_chunk]
                data_queue.put(chunk)

        return bytes_read
        
    @property   
    def ms_per_chunk(self) -> int:
        return self._ms_per_chunk
    
    @cached_property
    def bytes_per_chunk(self) -> int:
        sample_time = 1 / self.sample_rate
        sec_per_chunk = self._ms_per_chunk / 1000
        num_samples = sec_per_chunk / sample_time

        print("num_samples: ")
        print(num_samples)
        
        print("bytes_per_chunk: ")
        print(int(num_samples * 2)) # 2 bytes per sample
        
        return int(num_samples * 2)
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    