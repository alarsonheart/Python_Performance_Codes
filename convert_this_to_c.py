'''Takes code from from_wav_to_pcm_2 and from_pcm_to_fft.py and combines them into one program'''
import numpy as np
from scipy.fft import rfft, rfftfreq

# Constants
SAMPLE_RATE = 192000  # in Hz
PCM_BUFFER_SIZE = 4032  # Size of PCM_Buffer
DETECTION_FREQUENCY = 6000  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 47.5  # Tolerance for frequency detection (in Hz) --bin size
MAGNITUDE_THRESHOLD = 2000000  # Magnitude threshold for detection (singles out fundamentals instead of harmonics)

def find_top_frequencies(buffer):
    # Perform FFT
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)
    
    # Find frequencies
    freqs = rfftfreq(len(buffer), 1 / SAMPLE_RATE)
        
    # Get top frequencies and magnitudes
    top_indices = np.argsort(fft_magnitude)[-3:]
    top_frequencies = freqs[top_indices]
    top_magnitudes = fft_magnitude[top_indices]
    return top_frequencies, top_magnitudes

def main():
    detected_frequency = None

    while True:

        PCM_Buffer = []

        top_frequencies, top_magnitudes = find_top_frequencies(PCM_Buffer)

        for freq, mag in zip(top_frequencies, top_magnitudes):
            print(f"Frequency: {freq} Hz, Magnitude: {mag}")

        for freq, mag in zip(top_frequencies, top_magnitudes):
            if abs(freq - DETECTION_FREQUENCY) <= FREQUENCY_TOLERANCE and mag > MAGNITUDE_THRESHOLD:
                detected_frequency = freq
                break  # Exit the loop if a matching frequency with sufficient magnitude is found

        if detected_frequency is not None:
            break  # Exit the main loop if a matching frequency is found

    if detected_frequency is not None:
        print(f"{detected_frequency} Hz found")

main()