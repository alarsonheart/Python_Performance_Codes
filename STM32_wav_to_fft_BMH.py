
import numpy as np
from scipy.fft import rfft, rfftfreq
import time
import wave
import struct
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# Constants
SAMPLING_FREQUENCY = 188416  # in Hz (ADJUSTED FROM 192000 TO MATCH BIN SIZE OF 46 in STM32)
DETECTION_FREQUENCY = 1000  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 50  # Tolerance for frequency detection (in Hz)
TIMEOUT_SECONDS = 5  # Timeout in seconds

def apply_blackman_harris_window(data):
    window = np.blackman(len(data))
    return data * window

'''Calculates the PCM values of 21ms of data and zero-pads the last indicies since a buffer of 21ms is 4032'''
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

'''Takes each 4096 PCM Buffer (21ms) from read_pcm_frames and computes its FFT & finds the top 10 frequencies present'''
def find_top_frequencies(buffer, min_magnitude=10):
    # Apply Blackman-Harris window
    buffer = apply_blackman_harris_window(buffer)
    # Perform FFT
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)
    freqs = rfftfreq(len(buffer), 1 / SAMPLING_FREQUENCY) # Find frequencies
    top_indices = np.argsort(fft_magnitude)[-10:][::-1]
    top_frequencies = []
    top_magnitudes = []
    for idx in top_indices:
        if fft_magnitude[idx] >= min_magnitude:
            top_frequencies.append(freqs[idx])
            top_magnitudes.append(fft_magnitude[idx])
    return top_frequencies, top_magnitudes  # Return both frequencies and magnitudes

def visualize_fft_for_entire_clip(pcm_values, max_frequency=20000, height=None, distance=None):
    freqs = rfftfreq(len(pcm_values), 1 / SAMPLING_FREQUENCY)
    fft_result = rfft(pcm_values)
    fft_magnitude = np.abs(fft_result)
    mask = freqs <= max_frequency     # Limit the frequency range to a maximum of 20,000 Hz
    freqs = freqs[mask]
    fft_magnitude = fft_magnitude[mask]
    peaks, _ = find_peaks(fft_magnitude, height=6000000, distance=10000, threshold=20000)

    peak_freqs = freqs[peaks]
    peak_mags = fft_magnitude[peaks]
# Print the peak frequencies and their magnitudes
    print("Peak Frequencies and Magnitudes:")
    for freq, mag in zip(peak_freqs, peak_mags):
        print(f"Frequency: {freq} Hz, Magnitude: {mag}")

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude)
    plt.plot(freqs[peaks], fft_magnitude[peaks], "x")  # Mark the peaks
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Analysis for Entire Audio Clip (Up to 20,000 Hz)')
    plt.grid(True)
    plt.show()
    return peak_freqs, peak_mags

def process_audio_in_intervals(wav_file, interval_duration):
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()

    num_frames_per_interval = int(interval_duration * frame_rate)
    total_frames = wav_file.getnframes()
    intervals = total_frames // num_frames_per_interval
    unique_top_frequencies = {}
    unique_top_magnitudes = {}
    for _ in range(intervals + 1):
        pcm_values = read_pcm_frames(wav_file, num_frames_per_interval)
        top_frequencies, top_magnitudes = find_top_frequencies(pcm_values)
        for freq, mag in zip(top_frequencies, top_magnitudes):
            if freq not in unique_top_frequencies or mag > unique_top_magnitudes[freq]:
                unique_top_frequencies[freq] = mag
                unique_top_magnitudes[freq] = mag

            '''This prints the top 10 frequencies for each interval'''
        for freq, mag in zip(top_frequencies, top_magnitudes):
            print(f"Frequency: {freq} Hz, Magnitude: {mag}")
    '''This prints the top 10 frequencies found in the entire FFT (based only on magnitude so this is wrong)'''
    sorted_freq_mags = sorted(unique_top_frequencies.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 Frequencies and Magnitudes (STM32 Match):") #this prints the frequencies using the bin size & buffer process similar to how it is done in STM32
    for freq, mag in sorted_freq_mags[:10]:
        print(f"Frequency: {freq} Hz, Magnitude: {mag}")
    print()
    print()

def main():
    input_file_path = r"Audio_Files\500Hz_IMP23ABSU_MIC.wav"
    # input_file_path = r"Audio_Files\500_to_3340_IMP23ABSU.wav"
    # input_file_path = r"Audio_Files\400_1000_1700.wav"
    # input_file_path = r"Well_Audio\Well_1\pump_discharge_inner.wav"


    interval_duration = 0.021  # Define the duration in seconds (21ms)
    with wave.open(input_file_path, 'rb') as wav_file:
        process_audio_in_intervals(wav_file, interval_duration)

        '''This rewinds the .wav file and shows the fft for the entire audio clip'''
        wav_file.rewind()
        pcm_values = read_pcm_frames(wav_file, wav_file.getnframes())
        visualize_fft_for_entire_clip(pcm_values)
        

if __name__ == "__main__":
    main()
