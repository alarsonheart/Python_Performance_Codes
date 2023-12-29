import wave
import struct
import numpy as np
from scipy.fft import rfft, rfftfreq
import time

# Constants
input_file_path = "500_to_3340_IMP23ABSU.wav" 
duration_seconds = 0.021
SAMPLING_FREQUENCY = 192000  # in Hz
BUFFER_SIZE = 4032  # Size of PCM_Buffer
DETECTION_FREQUENCY = 500  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 50  # Tolerance for frequency detection (in Hz)
TIMEOUT_SECONDS = 5  # Timeout in seconds

def find_top_frequencies(buffer):
    # Perform FFT
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)
    
    # Find frequencies
    freqs = rfftfreq(len(buffer), 1 / SAMPLING_FREQUENCY)
    #size is half of the length of the buffer (n/2) so 2017
        
    # Get top 3 frequencies
    top_indices = np.argsort(fft_magnitude)[-3:]
    top_frequencies = freqs[top_indices]
    top_magnitudes = fft_magnitude[top_indices]
    return top_frequencies, top_magnitudes

def main():
    start_time = time.time()

    with wave.open(input_file_path, 'rb') as wav_file:
        num_frames = int(duration_seconds * SAMPLING_FREQUENCY)

        while True:
            frames = wav_file.readframes(num_frames)
            if len(frames) < num_frames * 2:  # 2 bytes per frame (16-bit PCM)
                break  # End of file

            pcm_values = np.array(struct.unpack(f"{len(frames) // 2}h", frames))

            top_frequencies, top_magnitudes = find_top_frequencies(pcm_values)

            for freq, mag in zip(top_frequencies, top_magnitudes):
                print(f"Frequency: {freq} Hz, Magnitude: {mag}")

            if any(abs(freq - DETECTION_FREQUENCY) <= FREQUENCY_TOLERANCE for freq in top_frequencies):
                print(f"{DETECTION_FREQUENCY}Hz found")
                break

            if time.time() - start_time >= TIMEOUT_SECONDS:
                print("Timeout reached. Target frequency not found.")
                break

main()
