import matplotlib.pyplot as plt
from scipy.signal import spectrogram, resample
from scipy.io import wavfile
import numpy as np

# Read the audio file
sample_rate, samples = wavfile.read('IMP23ABSU_MIC_1k_Test_Tone.wav')

# Downsample the audio to a lower sampling rate (e.g., 48kHz)
new_sample_rate = 48000
samples = resample(samples, int(len(samples) * (new_sample_rate / sample_rate)))

# Compute the spectrogram
frequencies, times, spectrogram = spectrogram(samples, new_sample_rate)
plt.pcolormesh(times, frequencies, spectrogram)

# Find the top 3 peak frequencies and their magnitudes
num_peaks = 3
peak_indices = np.argpartition(spectrogram, -num_peaks, axis=None)[-num_peaks:]
peak_indices = np.unravel_index(peak_indices, spectrogram.shape)
top_frequencies = frequencies[peak_indices[0]]
top_magnitudes = spectrogram[peak_indices]

for i in range(num_peaks):
    print(f"Top Peak {i + 1}: Frequency = {top_frequencies[i]:.2f} Hz, Magnitude = {top_magnitudes[i]:.2f}")

plt.show()

