import numpy as np
from scipy.fft import rfft, rfftfreq
import time
import wave
import struct
import matplotlib.pyplot as plt

# Constants
SAMPLING_FREQUENCY = 188416  # in Hz (ADJUSTED FROM 192000 TO MATCH BIN SIZE OF 46 in STM32)
DETECTION_FREQUENCY = 1000  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 50  # Tolerance for frequency detection (in Hz)
# Bin Size is 47.619 so 50 should work (or a little less--maybe exactly 47.619)
TIMEOUT_SECONDS = 5  # Timeout in seconds

def apply_blackman_harris_window(data):
    window = np.blackman(len(data))
    return data * window

def find_top_frequencies(buffer):
    # Apply Blackman-Harris window
    buffer = apply_blackman_harris_window(buffer)
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)

    # Find frequencies
    freqs = rfftfreq(len(buffer), 1 / SAMPLING_FREQUENCY)

    # Get top 3 frequencies
    top_indices = np.argsort(fft_magnitude)[-10:][::-1]
    top_frequencies = freqs[top_indices]
    top_magnitudes = fft_magnitude[top_indices]
    return top_frequencies, top_magnitudes  # Return both frequencies and magnitudes

def read_pcm_frames(wav_file, num_frames):
    pcm_values = []
    buffer_size = 4096
    while num_frames > 0:
        chunk_size = min(buffer_size, num_frames)
        frames = wav_file.readframes(chunk_size)
        pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
        pcm_values.extend(pcm_chunk)

        # If the chunk is smaller than buffer_size, pad with zeros
        if len(pcm_chunk) < buffer_size:
            pcm_values.extend([0] * (buffer_size - len(pcm_chunk)))

        num_frames -= chunk_size
    return pcm_values

def process_audio_in_intervals(wav_file, interval_duration):
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()

    num_frames_per_interval = int(interval_duration * frame_rate)
    total_frames = wav_file.getnframes()
    intervals = total_frames // num_frames_per_interval

    for _ in range(intervals + 1):
        pcm_values = read_pcm_frames(wav_file, num_frames_per_interval)

        # Find top frequencies and their magnitudes for this interval
        top_frequencies, top_magnitudes = find_top_frequencies(pcm_values)

        # Print the top frequencies and their magnitudes for this interval
        for freq, mag in zip(top_frequencies, top_magnitudes):
            print(f"Frequency: {freq} Hz, Magnitude: {mag}")

def visualize_fft_for_entire_clip(pcm_values, max_frequency=20000):
    freqs = rfftfreq(len(pcm_values), 1 / SAMPLING_FREQUENCY)
    fft_result = rfft(pcm_values)
    fft_magnitude = np.abs(fft_result)

    # Limit the frequency range to a maximum of 20,000 Hz
    mask = freqs <= max_frequency
    freqs = freqs[mask]
    fft_magnitude = fft_magnitude[mask]

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Analysis for Entire Audio Clip (Up to 20,000 Hz)')
    plt.grid(True)
    plt.show()



def main():
    # Input WAV file path
    input_file_path = r"Audio_Files\12311Hz_close_IMP23ABSU_MIC.wav"

    # Define the duration in seconds (21ms)
    interval_duration = 0.021

    # Open the WAV file for reading
    with wave.open(input_file_path, 'rb') as wav_file:
        # Get the audio parameters
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()

        print(f"Channels: {num_channels}")
        print(f"Sample Width: {sample_width * 8} bits")
        print(f"Frame Rate: {frame_rate} Hz")

        # Process audio in intervals
        process_audio_in_intervals(wav_file, interval_duration)

        # Rewind the WAV file to the beginning
        wav_file.rewind()

        # Read the entire audio clip
        pcm_values = read_pcm_frames(wav_file, wav_file.getnframes())

        # Visualize the FFT spectrum for the entire audio clip
        visualize_fft_for_entire_clip(pcm_values)

if __name__ == "__main__":
    main()
