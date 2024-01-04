'''Takes code from from_wav_to_pcm_2 and from_pcm_to_fft.py and combines them into one program'''
import wave
import struct
import numpy as np
from scipy.fft import rfft, rfftfreq
import time

# Constants
input_file_path = r"Audio_Files\500Hz_IMP23ABSU_MIC.wav" 
duration_seconds = 0.021
SAMPLING_FREQUENCY = 192000  # in Hz
BUFFER_SIZE = 4032  # Size of PCM_Buffer
DETECTION_FREQUENCY = 500  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 47.5  # Tolerance for frequency detection (in Hz) --bin size
MAGNITUDE_THRESHOLD = 2000000  # Magnitude threshold for detection (singles out fundamentals instead of harmonics)
TIMEOUT_SECONDS = 5  # Timeout in seconds

def find_top_frequencies(buffer):
    # Perform FFT
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)
    
    # Find frequencies
    freqs = rfftfreq(len(buffer), 1 / SAMPLING_FREQUENCY)
        
    # Get top frequencies and magnitudes
    top_indices = np.argsort(fft_magnitude)[-3:]
    top_frequencies = freqs[top_indices]
    top_magnitudes = fft_magnitude[top_indices]
    return top_frequencies, top_magnitudes

def main():
    start_time = time.time()
    detected_frequency = None

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

            for freq, mag in zip(top_frequencies, top_magnitudes):
                if abs(freq - DETECTION_FREQUENCY) <= FREQUENCY_TOLERANCE and mag > MAGNITUDE_THRESHOLD:
                    detected_frequency = freq
                    break  # Exit the loop if a matching frequency with sufficient magnitude is found

            if detected_frequency is not None:
                break  # Exit the main loop if a matching frequency is found

            if time.time() - start_time >= TIMEOUT_SECONDS:
                print("Timeout reached. Target frequency not found.")
                break

    if detected_frequency is not None:
        print(f"{detected_frequency} Hz found")

main()
