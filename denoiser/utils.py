import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(x, y, fs, label_x, label_y, domain='time', frame_size=None):
    if domain == 'time':
        plt.subplot(2, 1, 1)
        plot_time_domain(x, fs, label_x)
        
        plt.subplot(2, 1, 2)
        plot_time_domain(y, fs, label_y)
        
        plt.show()
    elif domain == 'freq':
        if frame_size is None:
            raise ValueError('frame_size must be specified for frequency domain plots')
        
        plot_freq_domain(x, fs, frame_size=frame_size, label=label_x)
        plot_freq_domain(y, fs, frame_size=frame_size, label=label_y)
        plt.show()

def plot_time_domain(x, fs, label):
    t = np.arange(len(x)) / fs
    
    plt.title(f'{label} (Time Domain)')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.plot(t, x, label=label)
    plt.legend()
    plt.show()
    
def plot_freq_domain(X, fs, frame_size, label, psd=False):
    plt.title(f'{label} (Frequency Domain)')
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    
    # freqs = fftfreq(len(window), 1 / self.sample_rate)[:self.frame_size]
    
    if psd:
        f, Pxx = plt.psd(X, NFFT=frame_size, Fs=fs, label=label)
        Pxx = 10 * np.log10(Pxx)
        
        plt.ylabel('Power Spectral Density [dB/Hz]')
        plt.plot(f, Pxx)
        
    else:
        f = np.arange(frame_size//2) * fs / frame_size
        amplitude_spectrum = np.abs(X[:frame_size//2])
        
        plt.ylabel('Amplitude')
        plt.plot(f, amplitude_spectrum)
        
    plt.legend()
    plt.show()
    
    
def plot_banded(X):
    plt.plot(X)
    plt.xlabel('Band Number')
    plt.grid(True)

    plt.show()
    
# def play_audio(waveform):
#     '''Play audio file'''
    
#     if isinstance(waveform, torch.Tensor):
#         waveform = waveform.numpy()
        
#     num_channels, num_frames = waveform.shape
    
#     if num_channels == 1:
        
#         if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
#             print("Warning: NaN or infinite value encountered in waveform")
#             waveform = np.nan_to_num(waveform)  # replace NaNs and infinite values with finite numbers

#         # Play audio
#         display(Audio(waveform, rate=SAMPLE_RATE))
#     else:
#         raise ValueError("Waveforms with more than one channel are not supported.")
    