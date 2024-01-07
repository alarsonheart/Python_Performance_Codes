import numpy as np
import wave
import struct
from scipy.signal.windows import blackmanharris
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks

# Constants
SAMPLING_FREQUENCY = 188416  # Adjusted sampling frequency
FRAME_SIZE = 4096            # Buffer size after zero-padding
VALID_DATA_SIZE = 4032       # Original buffer size before zero-padding
TIME_INTERVAL = 21 / 1000    # 21ms in seconds

# Function to read and zero-pad audio data
def read_pcm_frames(wav_file, num_frames):
    pcm_values = []
    while num_frames > 0:
        chunk_size = min(VALID_DATA_SIZE, num_frames)
        frames = wav_file.readframes(chunk_size)
        pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
        pcm_values.extend(pcm_chunk)

        # Zero-pad if necessary
        if len(pcm_chunk) < VALID_DATA_SIZE:
            pcm_values.extend([0] * (FRAME_SIZE - VALID_DATA_SIZE))

        num_frames -= chunk_size
    return np.array(pcm_values, dtype=np.float32)  # Ensure pcm_values are floating point

# Short Time Fourier Transform (STFT) with Blackman-Harris window
def stft(signal, frame_size, window_fn=blackmanharris):
    window = window_fn(frame_size)
    hop_size = frame_size // 2  # 50% overlap
    frames = stride_tricks.as_strided(signal, shape=(int((len(signal) - frame_size) / hop_size + 1), frame_size),
                                      strides=(signal.strides[0] * hop_size, signal.strides[0])).copy()
    frames *= window  # Apply window to each frame
    return np.fft.rfft(frames)

# Plotting the spectrogram
def plot_spectrogram(spectrogram, fs, title='Spectrogram', save_path=None):
    plt.figure(figsize=(12, 6))
    plt.imshow(20*np.log10(np.abs(spectrogram.T)), aspect='auto', origin='lower', 
               extent=[0, spectrogram.shape[0], 0, fs/2], cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Finding top unique peak frequencies
def find_top_unique_peak_frequencies(spectrogram, fs, frame_size, num_peaks=5):
    # Get frequency bins
    freqs = np.fft.rfftfreq(frame_size, d=1./fs)
    # Get the magnitude of the spectrogram
    magnitudes = np.abs(spectrogram)
    # Flatten the spectrogram for peak finding
    flattened_magnitudes = magnitudes.flatten()
    # Find peaks
    peaks, _ = find_peaks(flattened_magnitudes, height=0)

    # Find top unique frequencies
    top_frequencies = []
    for peak in sorted(peaks, key=lambda x: flattened_magnitudes[x], reverse=True):
        freq = freqs[peak % magnitudes.shape[1]]
        if freq not in top_frequencies:
            top_frequencies.append(freq)
        if len(top_frequencies) == num_peaks:
            break

    return top_frequencies

# Main function to process the audio file and plot the spectrogram
def process_audio_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        pcm_data = read_pcm_frames(wav_file, wav_file.getnframes())
        spectrogram = stft(pcm_data, FRAME_SIZE)

        # Plotting the spectrogram
        plot_spectrogram(spectrogram, SAMPLING_FREQUENCY, title='Audio Spectrogram')

        # Finding the top 5 unique peak frequencies
        top_frequencies = find_top_unique_peak_frequencies(spectrogram, SAMPLING_FREQUENCY, FRAME_SIZE)

        print("Top 5 Unique Peak Frequencies:")
        for freq in top_frequencies:
            print(f"{freq} Hz")

if __name__ == "__main__":
    audio_file_path = r"Audio_Files\500Hz_IMP23ABSU_MIC.wav"  # Replace with the actual audio file path
    process_audio_file(audio_file_path)
