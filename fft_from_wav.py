'''
Reads the audio file and extracts essential information, including the sample rate and audio samples.
Specifies a user-defined duration for analysis.
Normalizes the audio data to ensure consistent amplitude levels.
Performs a Fast Fourier Transform (FFT) on the audio data to convert it from the time domain to the frequency domain, revealing its frequency components.
Identifies and prints the top N peak frequencies and their corresponding magnitudes in the audio's frequency spectrum.
Visualizes the FFT magnitude spectrum of the audio data using a plot.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import find_peaks

# Replace with the path to your .wav file
# C:\Users\angel\OneDrive - Heartland Ag Tech\HATWellAcquision\Well_Data_Collected_8_31_2023\Good_Wells_8_31_2023\well_1\pump_discharge_inner_Exported\IMP23ABSU_MIC.wav
# C:\Users\angel\OneDrive - Heartland Ag Tech\HATWellAcquision\Python_Performance_Codes
input_wav_file = r'..\Well_Data_Collected_8_31_2023\Good_Wells_8_31_2023\well_1\below_motor_mount_Exported\IMP23ABSU_MIC.wav'
# input_wav_file = r'IMP23ABSU_MIC_1k_Test_Tone.wav'
# Read the .wav file
sample_rate, audio_data = wavfile.read(input_wav_file)

print(f" sample_rate {sample_rate}")

# Define the duration in seconds that you want to process
duration_to_process = 60  # seconds
#Note: if you want smaller frequency bins (i.e. more accurate frequencies), you increase the duration

# Calculate the number of samples to keep for the specified duration
num_samples_to_keep = int(duration_to_process * (sample_rate/60))

# Slice the audio data to keep only the specified duration
audio_data = audio_data[:num_samples_to_keep]

# Normalize the audio data based on the actual data range
audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
#converts to floating point 32 numbers, makes the data positive, and divides the data by the maximum amplitude in the audio data to normalize it from [0 to 1] (unsigned)

fft_size = 4096 #set the desired FFT size -- 4096 matches the current STM32


# Perform FFT on the audio data
fft_output = fft(audio_data, n=fft_size)

# Normalize the magnitudes by dividing by the fft_size
# normalized_magnitudes = np.abs(fft_output) / fft_size



# Calculate the corresponding frequencies for the FFT output
frequency_resolution = sample_rate / fft_size #this decides how big the frequency bins are
frequencies = np.arange(0, fft_size) * frequency_resolution #converts the index of the frequency bins to their frequency (in Hz)

# Find the top N peak frequencies and their corresponding magnitudes
num_peaks = 10  # Change this to the desired number of top peaks
peak_indices, peak_properties = find_peaks(np.abs(fft_output[:fft_size // 2]), height=0.1)
#absolute value converts from complex to real numbers, height is the minimum value for a peak to be considered a peak
#Note: peak height = peak magnitude, so if you are looking to find values over a certain magnitude, change height=
#Note: we analyze the first half of the fft due to symmetry and since the second half of the fft is the abs(complex conjugate) of the values

if len(peak_indices) < num_peaks:
    num_peaks = len(peak_indices)  # Set num_peaks to the actual number of peaks

top_peak_indices = peak_indices[np.argsort(-np.abs(fft_output[peak_indices]))[:num_peaks]]
top_peak_frequencies = frequencies[top_peak_indices]
top_peak_magnitudes = np.abs(fft_output[top_peak_indices])
# top_peak_heights = peak_properties["peak_heights"][np.argsort(-np.abs(fft_output[peak_indices]))][:num_peaks]  # Capture peak heights

# Create a pandas DataFrame to store the data
data = {'Frequency (Hz)': top_peak_frequencies, 'Magnitude': top_peak_magnitudes}
df = pd.DataFrame(data)

# Save the data to an Excel file
output_excel_file = 'output_data.xlsx'
df.to_excel(output_excel_file, index=False)

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
    print('')
    # print(f"Top Peak {i + 1} height (Audio Data): {top_peak_heights[i]:.2f}")

print(f"Data saved to {output_excel_file}")