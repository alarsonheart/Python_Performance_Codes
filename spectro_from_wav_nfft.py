import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import wavfile
import numpy as np
# Read the audio file
sample_rate, samples = wavfile.read('IMP23ABSU_MIC_1k_Test_Tone.wav')

print(sample_rate)

# Set the desired upper frequency limit (e.g., 2kHz)
upper_freq_limit = 5000

# Compute the spectrogram with the desired frequency range
frequencies, times, spectrogram = spectrogram(samples, sample_rate, nfft=4096)
#nfft the number of FFT points to calculate the frequenct content (controls the bin size)
plt.pcolormesh(times, frequencies[frequencies <= upper_freq_limit], spectrogram[frequencies <= upper_freq_limit])

# Find the top 3 peak frequencies and their magnitudes
num_peaks = 3
top_peak_indices = spectrogram[frequencies <= upper_freq_limit].argsort(axis=None)[-num_peaks:]
top_peak_indices = np.unravel_index(top_peak_indices, spectrogram[frequencies <= upper_freq_limit].shape)
top_frequencies = frequencies[frequencies <= upper_freq_limit][top_peak_indices[0]]
top_magnitudes = spectrogram[frequencies <= upper_freq_limit][top_peak_indices]

for i in range(num_peaks):
    print(f"Peak {i + 1}: Frequency = {top_frequencies[i]:.2f} Hz, Magnitude = {top_magnitudes[i]:.2f}")

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