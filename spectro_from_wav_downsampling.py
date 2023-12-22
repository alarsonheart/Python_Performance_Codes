'''This is more congruent with fft_from_wav.py'''
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, resample
from scipy.io import wavfile
import numpy as np

# Read the audio file
sample_rate, samples = wavfile.read(
    r'..\Well_Data_Collected_8_31_2023\Good_Wells_8_31_2023\well_1\below_motor_mount_Exported\IMP23ABSU_MIC.wav')

# Downsample the audio to a lower sampling rate (e.g., 48kHz)
new_sample_rate = 48000
samples = resample(samples, int(len(samples) * (new_sample_rate / sample_rate)))

# Compute the spectrogram
frequencies, times, spectrogram = spectrogram(samples, new_sample_rate)
plt.pcolormesh(times, frequencies, spectrogram)

# Find the top 10 peak frequencies and their magnitudes
num_peaks = 10  # Change this to the desired number of top peaks

# Create a dictionary to store frequencies and magnitudes
peak_dict = {}

for i in range(num_peaks):
    peak_indices = np.unravel_index(np.argmax(spectrogram), spectrogram.shape)
    freq = frequencies[peak_indices[0]]
    mag = spectrogram[peak_indices]

    # Skip if frequency is already in the dictionary
    while freq in peak_dict:
        spectrogram[peak_indices] = 0  # Zero out the repeated peak
        peak_indices = np.unravel_index(np.argmax(spectrogram), spectrogram.shape)
        freq = frequencies[peak_indices[0]]
        mag = spectrogram[peak_indices]

    peak_dict[freq] = mag

# Sort the dictionary by magnitude and get the top 10 frequencies
sorted_peaks = sorted(peak_dict.items(), key=lambda x: x[1], reverse=True)[:num_peaks]

for i, (freq, mag) in enumerate(sorted_peaks):
    print(f"Top Peak {i + 1}: Frequency = {freq:.2f} Hz, Magnitude = {mag:.2f}")

plt.show()

# Calculate the bin size
num_frequency_bins = len(frequencies)
bin_size_hz = (new_sample_rate / 2) / num_frequency_bins

print(f"Bin Size: {bin_size_hz:.2f} Hz")
