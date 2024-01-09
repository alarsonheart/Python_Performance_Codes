'''fft_for_21ms_test.py by Angela Larson'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import wave
import struct
from scipy.signal import find_peaks

# Constants
SAMPLING_FREQUENCY = 188416  # in Hz
CHUNK_SIZE = 4096
OVERLAP_SIZE = CHUNK_SIZE - 4032  # Adjust for 21ms of data at the given sampling frequency
MAX_FREQUENCY = 20000  # Maximum frequency of interest

def apply_blackman_harris_window(data):
    window = np.blackman(len(data))
    return data * window

def read_pcm_frames(wav_file, frame_size, overlap_size):
    # Calculate number of chunks with overlap
    num_chunks = (wav_file.getnframes() - overlap_size) // (frame_size - overlap_size)
    pcm_values = []
    
    # Read the initial frames to start the process
    frames = wav_file.readframes(frame_size)
    pcm_values.extend(struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames))

    for _ in range(1, num_chunks):
        # Move back by the overlap size
        wav_file.setpos(wav_file.tell() - overlap_size)
        # Read frames with overlap
        frames = wav_file.readframes(frame_size)
        pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
        # Discard the overlapped initial values and extend the remaining
        pcm_values.extend(pcm_chunk[overlap_size:])
        if _ == 1:
            print(f'pcm_values {pcm_values}')
    return pcm_values

def compute_ola_fft(wav_file, frame_size, overlap_size):
    pcm_values = read_pcm_frames(wav_file, frame_size, overlap_size)
    window = np.blackman(frame_size)
    fft_results = []

    for i in range(0, len(pcm_values) - frame_size + 1, frame_size - overlap_size):
        chunk = pcm_values[i:i + frame_size]
        windowed_chunk = chunk * window
        fft_result = rfft(windowed_chunk)
        fft_results.append(np.abs(fft_result))
        
    # The overlap-add method to combine FFT results
    full_fft = np.zeros_like(fft_results[0])
    for i, fft_result in enumerate(fft_results):
        full_fft[:len(fft_result)] += fft_result

    return full_fft

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
    overlap_size = 64  # Adjust based on your requirement

    with wave.open(input_file_path, 'rb') as wav_file:
        fft_magnitude = compute_ola_fft(wav_file, frame_size, overlap_size)
        freqs = rfftfreq(frame_size, 1 / SAMPLING_FREQUENCY)
        find_peaks_and_plot(freqs, fft_magnitude)


if __name__ == "__main__":
    main()