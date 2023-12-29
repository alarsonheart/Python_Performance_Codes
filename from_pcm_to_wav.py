'''This takes PCM data and converts it to .wav file
Uses: to see if the PCM data from OnBoardMics matches the PCM data from HSDatalog
Note: this code works when you run hsdatalog_data_export.py on the file (using IMP23ABSU)'''
import pandas as pd
import wave
import struct


def pcm_to_wav(csv_file_path, wav_file_path, sample_rate=192000):
    # Load the PCM data from the CSV file
    pcm_data = pd.read_csv(csv_file_path)

    # Assuming PCM data is in the 'MIC' column
    pcm_values = pcm_data['MIC'].values.astype('int16')

    # Parameters for the WAV file
    num_channels = 1  # Mono channel
    sample_width = 2  # 16-bit PCM
    frame_rate = sample_rate  # Sampling rate (frequency)
    num_frames = len(pcm_values)
    compression_type = "NONE"
    compression_name = "not compressed"

    # Create and set up the WAV file
    with wave.open(wav_file_path, 'w') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.setnframes(num_frames)
        wav_file.setcomptype(compression_type, compression_name)

        # Writing PCM data to the WAV file
        for value in pcm_values:
            wav_file.writeframes(struct.pack('<h', value))


# Usage
csv_file = 'IMP23ABSU_MIC_PCM_exported'  # Replace with your actual CSV file path
csv_file_path = csv_file + '.csv'
wav_file_path = csv_file + '.wav'  # Replace with your desired output WAV file path
pcm_to_wav(csv_file_path, wav_file_path)
