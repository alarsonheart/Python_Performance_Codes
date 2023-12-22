'''This somehow prints the values of magnitude at each time step, but is not accurate to fft_to_wav.py'''
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import wavfile
import numpy as np

# Read the audio file
sample_rate, samples = wavfile.read(
    r'..\Well_Data_Collected_8_31_2023\Good_Wells_8_31_2023\well_1\below_motor_mount_Exported\IMP23ABSU_MIC.wav')

# Set the desired upper frequency limit (e.g., 2kHz)
upper_freq_limit = 5000  # Hz

# Compute the spectrogram with the desired frequency range
frequencies, times, spectrogram = spectrogram(samples, sample_rate, nfft=4096)

# Find the top 10 peak frequencies and their magnitudes for each time slice
num_peaks = 10
peak_frequencies = []
peak_magnitudes = []

for i in range(spectrogram.shape[1]):
    spectrum = spectrogram[:, i]
    freq_indices = np.argsort(spectrum)[-num_peaks:]
    top_frequencies = frequencies[freq_indices]
    top_magnitudes = spectrum[freq_indices]

    # Create a dictionary to store frequencies and magnitudes for the current time slice
    time_slice_dict = {}

    for freq, mag in zip(top_frequencies, top_magnitudes):
        # Skip if frequency is already in the dictionary
        while freq in time_slice_dict:
            freq_indices = freq_indices[:-1]  # Remove the last index to get the next peak
            freq = frequencies[freq_indices[-1]]
            mag = spectrum[freq_indices[-1]]
        time_slice_dict[freq] = mag

    peak_frequencies.append(list(time_slice_dict.keys()))
    peak_magnitudes.append(list(time_slice_dict.values()))

# Plot the spectrogram
plt.pcolormesh(times, frequencies[frequencies <= upper_freq_limit], spectrogram[frequencies <= upper_freq_limit])

# Print the top peak frequencies and magnitudes for each time slice
for i in range(spectrogram.shape[1]):
    print(f"Time {times[i]:.2f} s:")
    for j in range(num_peaks):
        print(f"  Peak {j + 1}: Frequency = {peak_frequencies[i][j]:.2f} Hz, Magnitude = {peak_magnitudes[i][j]:.2f}")

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.show()

# from scipy import signal
# import matplotlib.pyplot as plt
# import numpy as np
#
# fs = 10e3 #sampling frequency -- THIS IS WHAT BOUNDS THE CODE -- Nyquist rate: 2 * max representable frequency
# N = 1e5 #number of samples in the signal
# amp = 2 * np.sqrt(2)  #amplitude of the carrier signal
# noise_power = 0.01 * fs / 2 #additional noise
# time = np.arange(N) / float(fs) #array that representst hte time axis (using sampling freq)
# mod = 500*np.cos(2*np.pi*0.25*time)  #this is our modulating signal with freq .25Hz and amplitude 500
# carrier = amp * np.sin(2*np.pi*3e3*time + mod) #modulated carrier signal (sin wave of 3000Hz)
# noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise *= np.exp(-time/5)
# x = carrier + noise #our final signal
# f, t, Sxx = signal.spectrogram(x, fs)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()