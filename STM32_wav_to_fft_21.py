import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import time
import wave
import struct
import matplotlib.pyplot  as plt

# Constants
SAMPLING_FREQUENCY = 188416  # in Hz (ADJUSTED FROM 192000 TO MATCH BIN SIZE OF 46 in STM32)
BUFFER_SIZE = 4096  # Size of PCM_Buffer
DETECTION_FREQUENCY = 1000  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 50  # Tolerance for frequency detection (in Hz)
# Bin Size is 47.619 so 50 should work (or a little less--maybe exactly 47.619)
TIMEOUT_SECONDS = 5  # Timeout in seconds

def wave_to_pcm():
    # Input WAV file path
    # input_file_path = r"Audio_Files\500_to_3340_IMP23ABSU.wav"
    input_file_path = r"Audio_Files\500_to_3340_IMP23ABSU.wav"


    # Define the duration in seconds (21ms)
    duration_seconds = 0.021

    # Open the WAV file for reading
    with wave.open(input_file_path, 'rb') as wav_file:
        # Get the audio parameters
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()

        print(f"Channels: {num_channels}")
        print(f"Sample Width: {sample_width * 8} bits")
        print(f"Frame Rate: {frame_rate} Hz")

        # Calculate the number of frames for the first 21ms
        num_frames = int(duration_seconds * frame_rate)

        pcm_values = []

        # Read audio frames and collect PCM values in chunks of 4096 samples
        buffer_size = 4096
        while num_frames > 0:
            chunk_size = min(buffer_size, num_frames)
            frames = wav_file.readframes(chunk_size)
            pcm_chunk = struct.unpack(f"{len(frames) // sample_width}h", frames)
            pcm_values.extend(pcm_chunk)

            # If the chunk is smaller than buffer_size, pad with zeros
            if len(pcm_chunk) < buffer_size:
                pcm_values.extend([0] * (buffer_size - len(pcm_chunk)))

            num_frames -= chunk_size

        # Return PCM values as a list
        return pcm_values

def find_top_frequencies(buffer):
    # Perform FFT
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)

    # Find frequencies
    freqs = rfftfreq(len(buffer), 1 / SAMPLING_FREQUENCY)

    # Get top 3 frequencies
    top_indices = np.argsort(fft_magnitude)[-10:][::-1]
    top_frequencies = freqs[top_indices]
    top_magnitudes = fft_magnitude[top_indices]
    return top_frequencies, top_magnitudes  # Return both frequencies and magnitudes

def visualize_fft_for_entire_clip(pcm_values, max_frequency=20000, height=None, distance=None):
    freqs = rfftfreq(len(pcm_values), 1 / SAMPLING_FREQUENCY)
    fft_result = rfft(pcm_values)
    fft_magnitude = np.abs(fft_result)
    mask = freqs <= max_frequency     # Limit the frequency range to a maximum of 20,000 Hz
    freqs = freqs[mask]
    fft_magnitude = fft_magnitude[mask]
    peaks, _ = find_peaks(fft_magnitude, height=60000, distance=10000, ) #threshold= 20000

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude)
    plt.plot(freqs[peaks], fft_magnitude[peaks], "x")  # Mark the peaks
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Analysis for Entire Audio Clip (Up to 20,000 Hz)')
    plt.grid(True)
    plt.show()
    peak_freqs = freqs[peaks]
    peak_mags = fft_magnitude[peaks]
    return peak_freqs, peak_mags

def main():
    buffer_data = wave_to_pcm()

    # Find top 3 frequencies and their magnitudes
    top_frequencies, top_magnitudes = find_top_frequencies(buffer_data)
    
    visualize_fft_for_entire_clip(buffer_data)
    # Print the top frequencies and their magnitudes
    for freq, mag in zip(top_frequencies, top_magnitudes):
        print(f"Frequency: {freq} Hz, Magnitude: {mag}")
    
        # Find frequencies
    # freqs = rfftfreq(len(buffer_data), 1 / SAMPLING_FREQUENCY)

    # fft_result = rfft(buffer_data)
    # fft_magnitude = np.abs(fft_result)
    
    # # Visualize the entire FFT spectrum
    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs, fft_magnitude)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('FFT Analysis')
    # plt.grid(True)
    # plt.show()

main()