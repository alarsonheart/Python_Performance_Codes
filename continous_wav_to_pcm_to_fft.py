'''This is a combo of from_wav_to_pcm2 and from_pcm_to_fft'''
import numpy as np
from scipy.fft import rfft, rfftfreq
import time
import wave
import struct

# Constants
SAMPLING_FREQUENCY = 192000  # in Hz
BUFFER_SIZE = 4032  # Size of PCM_Buffer
DETECTION_FREQUENCY = 20000  # Frequency to detect (in Hz)
FREQUENCY_TOLERANCE = 50  # Tolerance for frequency detection (in Hz)
TIMEOUT_SECONDS = 5  # Timeout in seconds

def find_top_frequencies(buffer):
    # Perform FFT
    fft_result = rfft(buffer)
    fft_magnitude = np.abs(fft_result)
    
    # Find frequencies
    freqs = rfftfreq(len(buffer), 1 / SAMPLING_FREQUENCY)
    
    # Get top 3 frequencies
    top_indices = np.argsort(fft_magnitude)[-3:]
    top_frequencies = freqs[top_indices]
    return top_frequencies

def process_pcm_data(pcm_data, PCM_Buffer):
    PCM_Buffer.extend(pcm_data)
    if len(PCM_Buffer) >= BUFFER_SIZE:
        return True
    else:
        return False

def from_wav_to_pcm(PCM_Buffer):
    # Input WAV file path
    input_file_path = "20_to_20k_audio_3.wav"

    # Open the WAV file for reading
    with wave.open(input_file_path, 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()

        while True:
            # Calculate the number of frames for the next 21ms
            duration_seconds = 0.021
            num_frames = int(duration_seconds * frame_rate)
            frames = wav_file.readframes(num_frames)

            # Break if no more frames are left
            if not frames:
                print("End of file reached.")
                break

            # Read audio frames and collect PCM values
            pcm_values = struct.unpack(f"{len(frames) // sample_width}h", frames)

            # Process PCM data and check for the target frequency
            if process_pcm_data(pcm_values, PCM_Buffer):
                top_frequencies = find_top_frequencies(PCM_Buffer)

                if any(abs(freq - DETECTION_FREQUENCY) <= FREQUENCY_TOLERANCE for freq in top_frequencies):
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    print(f"{DETECTION_FREQUENCY} Hz frequency found at {elapsed_time:.2f} seconds")
                    break  # Stop if the target frequency is found

                PCM_Buffer = PCM_Buffer[BUFFER_SIZE:]

            # Check for timeout
            if time.time() - start_time >= TIMEOUT_SECONDS:
                print("Timeout reached. Target frequency not found.")
                break

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    PCM_Buffer = []  # Initialize PCM_Buffer
    from_wav_to_pcm(PCM_Buffer)

