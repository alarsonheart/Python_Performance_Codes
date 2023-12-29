from scipy.io import wavfile
import numpy as np
import pandas as pd

# Read the input .wav file
sample_rate, sig = wavfile.read("IMP23ABSU_MIC_1k_Test_Tone.wav")

# Check if the file is already in 16-bit PCM format
if sig.dtype == np.int16:
    # Ensure that the audio data is in the desired range (0 to 65535)
    sig = ((sig - np.min(sig)) / (np.max(sig) - np.min(sig)) * 65535).astype(np.uint16)

    # Extract the first second of audio
    num_samples_to_extract = sample_rate  # Number of samples in 1 second
    sig_extracted = sig[:num_samples_to_extract]

    # Create a DataFrame from the PCM values
    df = pd.DataFrame({"PCM Values": sig_extracted})

    # Save the DataFrame to an Excel file
    df.to_excel("pcm_values.xlsx", index=False)

    print("PCM values of the first second saved to 'pcm_values.xlsx'.")
elif sig.dtype == np.float32:
    # Ensure that the audio data is in a float32 format
    # Rescale the audio data to 16-bit PCM (0 to 65535)
    sig = ((sig - np.min(sig)) / (np.max(sig) - np.min(sig)) * 65535).astype(np.uint16)

    # Extract the first second of audio
    num_samples_to_extract = sample_rate  # Number of samples in 1 second
    sig_extracted = sig[:num_samples_to_extract]

    # Create a DataFrame from the PCM values
    df = pd.DataFrame({"PCM Values": sig_extracted})

    # Save the DataFrame to an Excel file
    df.to_excel("pcm_values.xlsx", index=False)

    print("Audio data converted to 16-bit PCM (0 to 65535), and PCM values of the first second saved to 'pcm_values.xlsx'.")
else:
    print("Not 16-bit or 32-bit float")
