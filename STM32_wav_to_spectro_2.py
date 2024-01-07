import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
SAMPLING_FREQUENCY = 188416  # Hz, as specified
from numpy.lib import stride_tricks

FRAME_SIZE = 4096  # Total buffer size
valid_data_size = 4032  # Valid data points in the buffer
zero_padding = FRAME_SIZE - valid_data_size  # Number of points to be zero-padded
time_interval = 21 / 1000  # 21 ms in seconds

# Calculating the number of samples in 21ms
samples_in_21ms = int(SAMPLING_FREQUENCY * time_interval)
overlap_fac = 1 - (samples_in_21ms / FRAME_SIZE)  # Calculated overlap factor

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=overlap_fac, window=np.blackman):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by windows)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

def logscale_spec(spec, sr=188416, factor=20.):
    time_bins, freq_bins = np.shape(spec)
    scale = np.linspace(0, 1, freq_bins) ** factor
    scale *= (freq_bins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([time_bins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freq_bins*2, 1./sr)[:freq_bins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    samplerate = 188416
    s = stft(samples, FRAME_SIZE)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    # Find peaks in the spectrogram
    peak_freqs = []
    peak_magnitudes = []

    for t in range(timebins):
        for f in range(1, freqbins - 1):
            if ims[t, f] > ims[t, f - 1] and ims[t, f] > ims[t, f + 1]:
                peak_freqs.append(freq[f])
                peak_magnitudes.append(ims[t, f])

    # Create a dictionary to store unique peak frequencies and their magnitudes
    unique_peak_dict = {}
    for freq, mag in zip(peak_freqs, peak_magnitudes):
        if freq not in unique_peak_dict:
            unique_peak_dict[freq] = mag
        else:
            # If the frequency is repeated, choose the maximum magnitude
            if mag > unique_peak_dict[freq]:
                unique_peak_dict[freq] = mag

    # Sort unique peak frequencies by magnitude in descending order
    sorted_peak_freqs = sorted(unique_peak_dict, key=lambda k: unique_peak_dict[k], reverse=True)

    # Print the top 10 unique peak frequencies and their magnitudes
    print("Top 10 Unique Peak Frequencies and Magnitudes:")
    for freq in sorted_peak_freqs[:10]:
        print(f"Peak Frequency: {freq:.2f} Hz, Magnitude: {unique_peak_dict[freq]:.2f} dB")

    return ims, peak_freqs, peak_magnitudes

if __name__ == "__main__":
    # Example usage
    audio_path = r"Audio_Files\500Hz_IMP23ABSU_MIC.wav"
    plotstft(audio_path)
