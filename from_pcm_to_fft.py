import numpy as np
from scipy.fft import rfft, rfftfreq
import time

# Constants
SAMPLING_FREQUENCY = 192000  # in Hz
BUFFER_SIZE = 4032  # Size of PCM_Buffer
DETECTION_FREQUENCY = 1000  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 50  # Tolerance for frequency detection (in Hz)
#Bin Size is 47.619 so 50 should work (or a little less--maybe exactly 47.619)
TIMEOUT_SECONDS = 5  # Timeout in seconds

# Define your PCM buffer here
# Replace this with your actual PCM buffer data
PCM_Buffer = []
def find_top_frequencies(buffer):
    # Perform FFT
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)
    
    # Find frequencies
    freqs = rfftfreq(len(buffer), 1 / SAMPLING_FREQUENCY)
    
    # Get top 3 frequencies
    top_indices = np.argsort(fft_magnitude)[-3:]
    top_frequencies = freqs[top_indices]
    top_magnitudes = fft_magnitude[top_indices]
    return top_frequencies, top_magnitudes  # Return both frequencies and magnitudes

def main():
    start_time = time.time()  # Record the start time
    while True:
        # Use the PCM_Buffer directly
        buffer_data = PCM_Buffer

        # Find top 3 frequencies and their magnitudes
        top_frequencies, top_magnitudes = find_top_frequencies(buffer_data)
        
        # Print the top frequencies and their magnitudes
        for freq, mag in zip(top_frequencies, top_magnitudes):
            print(f"Frequency: {freq} Hz, Magnitude: {mag}")

        # Check if 1kHz is among the top frequencies
        if any(abs(freq - DETECTION_FREQUENCY) <= FREQUENCY_TOLERANCE for freq in top_frequencies):
            print("1kHz found")
            break  # Stop if 1kHz is found

        # Check for timeout
        if time.time() - start_time >= TIMEOUT_SECONDS:
            print("Timeout reached. Target frequency not found.")
            break

main()
