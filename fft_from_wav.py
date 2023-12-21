import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import find_peaks

# Replace 'your_audio_file.wav' with the path to your .wav file
input_wav_file = 'IMP23ABSU_MIC_1k_Test_Tone.wav'

# Read the .wav file
sample_rate, audio_data = wavfile.read(input_wav_file)

print(f" sample_rate {sample_rate}")

# Define the duration in seconds that you want to process
duration_to_process = .2133  # seconds

# Calculate the number of samples to keep for the specified duration
num_samples_to_keep = int(duration_to_process * sample_rate)

# Slice the audio data to keep only the specified duration
audio_data = audio_data[:num_samples_to_keep]

# Normalize the audio data based on the actual data range
audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))

# Perform FFT on the audio data
fft_output = fft(audio_data)

# Calculate the corresponding frequencies for the FFT output
fft_size = len(fft_output)
frequency_resolution = sample_rate / fft_size
frequencies = np.arange(0, fft_size) * frequency_resolution

# Find the top N peak frequencies and their corresponding magnitudes
num_peaks = 5  # Change this to the desired number of top peaks
peak_indices, _ = find_peaks(np.abs(fft_output[:fft_size // 2]), height=0.01)  # Adjust 'height' as needed
top_peak_indices = peak_indices[np.argsort(-np.abs(fft_output[peak_indices]))[:num_peaks]]
top_peak_frequencies = frequencies[top_peak_indices]
top_peak_magnitudes = np.abs(fft_output[top_peak_indices])

# Print the FFT Magnitude Spectrum
plt.figure()
plt.plot(frequencies[:fft_size // 2], np.abs(fft_output[:fft_size // 2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Magnitude Spectrum (Audio Data)')
plt.grid(True)
plt.show()

# Print the top N peak frequencies and their magnitudes
for i in range(num_peaks):
    print(f"Top Peak {i + 1} frequency (Audio Data): {top_peak_frequencies[i]:.2f} Hz")
    print(f"Top Peak {i + 1} magnitude (Audio Data): {top_peak_magnitudes[i]:.2f}")
