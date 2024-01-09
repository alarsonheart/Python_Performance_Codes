import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import wave
import struct
from scipy.signal import find_peaks

# Constants
SAMPLING_FREQUENCY = 188416  # in Hz
CHUNK_SIZE = 4096
DATA_SIZE = 4032  # Actual data size in each chunk
MAX_FREQUENCY = 20000  # Maximum frequency of interest

def apply_blackman_harris_window(data):
    window = np.blackman(len(data))
    return data * window

def read_and_pad_pcm_frames(wav_file, frame_size, data_size):
    pcm_values = []
    num_chunks = wav_file.getnframes() // data_size

    for _ in range(num_chunks):
        frames = wav_file.readframes(data_size)
        pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
        # Zero-pad the chunk to the frame_size
        padded_chunk = np.pad(pcm_chunk, (0, frame_size - len(pcm_chunk)), 'constant')
        pcm_values.extend(padded_chunk)

    return pcm_values

def compute_fft_with_zero_padding(wav_file, frame_size, data_size):
    pcm_values = read_and_pad_pcm_frames(wav_file, frame_size, data_size)
    window = np.blackman(frame_size)
    fft_results = []

    for i in range(0, len(pcm_values), frame_size):
        chunk = pcm_values[i:i + frame_size]
        windowed_chunk = chunk * window
        fft_result = rfft(windowed_chunk)
        fft_results.append(np.abs(fft_result))

    # Average the FFT results from all chunks
    average_fft = np.mean(fft_results, axis=0)
    return average_fft

def find_peaks_and_plot(freqs, fft_magnitude):
    # Apply the frequency mask
    mask = freqs <= MAX_FREQUENCY
    freqs = freqs[mask]
    fft_magnitude = fft_magnitude[mask]
    
    # Find the peaks
    peaks, _ = find_peaks(fft_magnitude)#, height=6000000, distance=10000, threshold=20000)
    peak_freqs = freqs[peaks]
    peak_mags = fft_magnitude[peaks]

    # Print the peak frequencies and their magnitudes
    print("Peak Frequencies and Magnitudes:")
    for freq, mag in zip(peak_freqs, peak_mags):
        print(f"Frequency: {freq} Hz, Magnitude: {mag}")

    # Plot the FFT and the peaks
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude, label='FFT Magnitude')
    plt.plot(peak_freqs, peak_mags, "x", label='Peaks')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT with Peaks for Entire Audio Clip (Up to 20,000 Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    input_file_path = r"Well_Audio\Well_4\pump_inner.wav"
    frame_size = 4096
    data_size = 4032  # Size of actual data in each chunk

    with wave.open(input_file_path, 'rb') as wav_file:
        fft_magnitude = compute_fft_with_zero_padding(wav_file, frame_size, data_size)
        freqs = rfftfreq(frame_size, 1 / SAMPLING_FREQUENCY)
        find_peaks_and_plot(freqs, fft_magnitude)

if __name__ == "__main__":
    main()
