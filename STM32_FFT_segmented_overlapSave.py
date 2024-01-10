import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
import wave
import struct
import matplotlib.pyplot as plt

# Constants
SAMPLING_FREQUENCY = 188416
DETECTION_FREQUENCY = 1000
FREQUENCY_TOLERANCE = 50
TIMEOUT_SECONDS = 5
PCM_CHUNK_LENGTH = 4032
FFT_BUFFER_SIZE = 4096

def apply_blackman_harris_window(data):
    window = np.blackman(len(data))
    return data * window

def overlap_save(pcm_values, filter_coeffs):
    buffer_size = FFT_BUFFER_SIZE
    overlap_size = min(len(filter_coeffs), buffer_size // 2)
    result = []
    overlap = np.zeros(overlap_size)

    for i in range(0, len(pcm_values), buffer_size - overlap_size):
        chunk = pcm_values[i:i + buffer_size]

        if len(chunk) < buffer_size:
            chunk += [0] * (buffer_size - len(chunk))

        chunk = apply_blackman_harris_window(chunk)
        chunk_fft = rfft(chunk)
        filtered_chunk_fft = chunk_fft * filter_coeffs
        filtered_chunk = irfft(filtered_chunk_fft)

        filtered_chunk_sliced = filtered_chunk[:buffer_size - overlap_size]
        if len(filtered_chunk_sliced) < buffer_size - overlap_size:
            filtered_chunk_sliced = np.concatenate((filtered_chunk_sliced, np.zeros(buffer_size - overlap_size - len(filtered_chunk_sliced))))

        result.extend(overlap + filtered_chunk_sliced)
        overlap = filtered_chunk[buffer_size - overlap_size:]

    return result

def read_pcm_frames(wav_file, num_frames):
    pcm_values = []
    buffer_size = FFT_BUFFER_SIZE

    while num_frames > 0:
        chunk_size = min(buffer_size, num_frames)
        frames = wav_file.readframes(chunk_size)
        pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
        pcm_values.extend(pcm_chunk)

        if len(pcm_chunk) < buffer_size:
            pcm_values.extend([0] * (buffer_size - len(pcm_chunk)))

        num_frames -= chunk_size

    return pcm_values




def modify_filter_coeffs(filter_coeffs):
    fft_output_size = (FFT_BUFFER_SIZE // 2) + 1
    if len(filter_coeffs) < fft_output_size:
        extended_filter_coeffs = np.zeros(fft_output_size)
        extended_filter_coeffs[:len(filter_coeffs)] = filter_coeffs
    else:
        extended_filter_coeffs = filter_coeffs[:fft_output_size]
    return extended_filter_coeffs

def process_audio_in_intervals(wav_file, interval_duration, filter_coeffs):
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()

    num_frames_per_interval = int(interval_duration * frame_rate)
    total_frames = wav_file.getnframes()
    intervals = total_frames // num_frames_per_interval

    all_filtered_pcm = []  # Collect all filtered PCM data here

    for _ in range(intervals + 1):
        pcm_values = read_pcm_frames(wav_file, num_frames_per_interval)
        filtered_pcm = overlap_save(pcm_values, filter_coeffs)
        all_filtered_pcm.extend(filtered_pcm)

    return all_filtered_pcm
def next_power_of_two(n):
    return 2 ** np.ceil(np.log2(n)).astype(int)
def main():
    input_file_path = r"Well_Audio\Well_4\pump_inner.wav"
    interval_duration = 0.021

    original_filter_coeffs = np.array([1.0] * 64)
    modified_filter_coeffs = modify_filter_coeffs(original_filter_coeffs)

    with wave.open(input_file_path, 'rb') as wav_file:
        all_filtered_pcm = process_audio_in_intervals(wav_file, interval_duration, modified_filter_coeffs)

    # Pad all_filtered_pcm to the next power of two
    padded_length = next_power_of_two(len(all_filtered_pcm))
    all_filtered_pcm_padded = np.pad(all_filtered_pcm, (0, padded_length - len(all_filtered_pcm)), 'constant')

    # Perform FFT on the padded filtered PCM data
    fft_result = rfft(all_filtered_pcm_padded)

    # Correctly calculate frequency axis based on the length of the padded data
    frequency_axis = rfftfreq(padded_length, 1 / SAMPLING_FREQUENCY)

    # Confirm lengths
    print(f"Length of all_filtered_pcm_padded: {len(all_filtered_pcm_padded)}")
    print(f"Length of fft_result: {len(fft_result)}")
    print(f"Length of frequency_axis: {len(frequency_axis)}")

    # Plot the FFT of the whole audio
    plt.figure()
    plt.plot(frequency_axis, np.abs(fft_result))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Entire Filtered Audio')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()






# '''STM32_wav_to_fft_BMH.py by Angela Larson'''

# import numpy as np
# from scipy.fft import rfft, rfftfreq
# import time
# import wave
# import struct
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# # Constants
# SAMPLING_FREQUENCY = 188416  # in Hz (ADJUSTED FROM 192000 TO MATCH BIN SIZE OF 46 in STM32)
# DETECTION_FREQUENCY = 1000  # Frequency to detect (in Hz)
# FREQUENCY_TOLERANCE = 50  # Tolerance for frequency detection (in Hz)
# TIMEOUT_SECONDS = 5  # Timeout in seconds
# PCM_CHUNK_LENGTH = 4032
# FFT_BUFFER_SIZE = 4096

# def apply_blackman_harris_window(data):
#     window = np.blackman(len(data))
#     return data * window

# '''Calculates the PCM values of 21ms of data and zero-pads the last indicies since a buffer of 21ms is 4032'''
# def read_pcm_frames(wav_file, num_frames):
#     pcm_values = []
#     buffer_size = FFT_BUFFER_SIZE
#     while num_frames > 0:
#         chunk_size = min(buffer_size, num_frames)
#         frames = wav_file.readframes(chunk_size)
#         pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames) #this is our 4032 length
#         pcm_values.extend(pcm_chunk)
        

#         # If the chunk is smaller than buffer_size, pad with zeros
#         if len(pcm_chunk) < buffer_size:
#             pcm_values.extend([0] * (buffer_size - len(pcm_chunk)))

#         num_frames -= chunk_size
#     return pcm_values

# def process_audio_in_intervals(wav_file, interval_duration):
#     sample_width = wav_file.getsampwidth()
#     frame_rate = wav_file.getframerate()

#     num_frames_per_interval = int(interval_duration * frame_rate)
#     total_frames = wav_file.getnframes()
#     intervals = total_frames // num_frames_per_interval
#     for _ in range(intervals + 1):
#         pcm_values = read_pcm_frames(wav_file, num_frames_per_interval)
#         #APPLY OLS HERE


        
# def main():
#     # input_file_path = r"Audio_Files\1300Hz_IMP23ABSU_MIC.wav"
#     # input_file_path = r"Audio_Files\500_to_3340_IMP23ABSU.wav"
#     # input_file_path = r"Audio_Files\400_1000_1700.wav"
#     input_file_path = r"Well_Audio\Well_4\pump_inner.wav"


#     interval_duration = 0.021  # Define the duration in seconds (21ms)
#     with wave.open(input_file_path, 'rb') as wav_file:
#         process_audio_in_intervals(wav_file, interval_duration)

# if __name__ == "__main__":
#     main()



'''import numpy as np
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

def read_pcm_frames(wav_file, num_frames):
    pcm_values = []
    while num_frames > 0:
        frames = wav_file.readframes(num_frames)
        pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
        pcm_values.extend(pcm_chunk)
        num_frames -= len(pcm_chunk)
    return pcm_values

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
    return top_frequencies, top_magnitudes

def process_audio_with_ols(wav_file, buffer_size, overlap_size):
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()

    num_frames = wav_file.getnframes()
    num_segments = (num_frames - buffer_size) // overlap_size + 1

    all_top_frequencies = []
    all_top_magnitudes = []

    for segment_index in range(num_segments):
        start_sample = segment_index * overlap_size
        end_sample = start_sample + buffer_size
        pcm_values = read_pcm_frames(wav_file, buffer_size)

        top_frequencies, top_magnitudes = find_top_frequencies(pcm_values)
        all_top_frequencies.append(top_frequencies)
        all_top_magnitudes.append(top_magnitudes)

    return all_top_frequencies, all_top_magnitudes

def main():
    # input_file_path = r"Well_Audio\Well_3\pump_inner.wav"
    input_file_path = r"500Hz_tone.wav"


    buffer_size = 4096
    overlap_size = 64  # Adjust this value as needed for your application

    with wave.open(input_file_path, 'rb') as wav_file:
        all_top_frequencies, all_top_magnitudes = process_audio_with_ols(wav_file, buffer_size, overlap_size)

    # Plot the FFT results
    all_top_frequencies = np.array(all_top_frequencies)
    all_top_magnitudes = np.array(all_top_magnitudes)

    plt.figure(figsize=(10, 6))
    plt.imshow(all_top_magnitudes.T, cmap='viridis', origin='lower', aspect='auto', extent=[0, len(all_top_magnitudes), 0, SAMPLING_FREQUENCY / 2])
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Segment Index')
    plt.ylabel('Frequency (Hz)')
    plt.title('Continuous FFT Analysis with Overlapping Segments')
    plt.show()

if __name__ == "__main__":
    main()
'''



# '''fft_for_21ms_test.py by Angela Larson'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import rfft, rfftfreq
# import wave
# import struct
# from scipy.signal import find_peaks

# # Constants
# SAMPLING_FREQUENCY = 188416  # in Hz (ADJUSTED FROM 192000 TO MATCH BIN SIZE OF 46 in STM32)
# CHUNK_SIZE = 4096
# OVERLAP_SIZE = CHUNK_SIZE - 4032  # Adjust for 21ms of data at the given sampling frequency
# MAX_FREQUENCY = 20000  # Maximum frequency of interest

# def apply_blackman_harris_window(data):
#     window = np.blackman(len(data))
#     return data * window

# def read_pcm_frames(wav_file, frame_size, overlap_size):
#     num_frames = wav_file.getnframes()
#     pcm_values = []

#     for i in range(0, num_frames - frame_size + 1, frame_size - overlap_size):
#         frames = wav_file.readframes(frame_size - overlap_size)
#         pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
#         pcm_values.extend(pcm_chunk)

#     # Handle the remaining frames (if any) by padding with zeros
#     remaining_frames = num_frames % (frame_size - overlap_size)
#     if remaining_frames > 0:
#         frames = wav_file.readframes(remaining_frames)
#         pcm_chunk = struct.unpack(f"{len(frames) // wav_file.getsampwidth()}h", frames)
#         pcm_values.extend(pcm_chunk)
#         pcm_values.extend([0] * (frame_size - overlap_size - remaining_frames))

#     return pcm_values


# def compute_ols_fft(wav_file, frame_size, overlap_size):
#     pcm_values = read_pcm_frames(wav_file, frame_size, overlap_size)
#     fft_results = []

#     for i in range(0, len(pcm_values), frame_size - overlap_size):
#         chunk = pcm_values[i:i + frame_size]
#         windowed_chunk = apply_blackman_harris_window(chunk)  # Apply the window
#         fft_result = rfft(windowed_chunk)
#         # Zero-pad the FFT result to make it consistent
#         if len(fft_result) < frame_size // 2 + 1:
#             fft_result = np.pad(fft_result, (0, frame_size // 2 + 1 - len(fft_result)))
#         fft_results.append(fft_result)

#     return fft_results  # Store FFT results for each frame in a list
# def find_peaks_and_plot(freqs, fft_results):
#     # Combine the FFT results by summing them
#     fft_magnitude = np.sum(np.abs(fft_results), axis=0)

#     # Apply the frequency mask
#     mask = freqs <= MAX_FREQUENCY
#     freqs = freqs[mask]
    
#     # Make sure fft_magnitude has the same shape as freqs
#     fft_magnitude = fft_magnitude[:len(freqs)]

#     # Find the peaks
#     peaks, _ = find_peaks(fft_magnitude, height=2500000)  # Modify as needed
#     peak_freqs = freqs[peaks]
#     peak_mags = fft_magnitude[peaks]

#     # Print the peak frequencies and their magnitudes
#     print("Peak Frequencies and Magnitudes:")
#     for freq, mag in zip(peak_freqs, peak_mags):
#         print(f"Frequency: {freq} Hz, Magnitude: {mag}")

#     # Plot the FFT and the peaks
#     plt.figure(figsize=(10, 6))
#     plt.plot(freqs, fft_magnitude, label='FFT Magnitude')
#     plt.plot(peak_freqs, peak_mags, "x", label='Peaks')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.title('FFT with Peaks for Entire Audio Clip (STM32_FFT_segmented_overlapSave.py)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def main():
#     input_file_path = r"Well_Audio\Well_4\pump_inner.wav"
#     frame_size = 4096
#     overlap_size = 64  # Adjust based on your requirement

#     with wave.open(input_file_path, 'rb') as wav_file:
#         fft_results = compute_ols_fft(wav_file, frame_size, overlap_size)
#         fft_magnitude = np.sum(np.abs(fft_results), axis=0)  # Sum FFT results for OLS
#         freqs = rfftfreq(len(fft_magnitude), 1 / SAMPLING_FREQUENCY)
#         find_peaks_and_plot(freqs, fft_results)

# if __name__ == "__main__":
#     main()


