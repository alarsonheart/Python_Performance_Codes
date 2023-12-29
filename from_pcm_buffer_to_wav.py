import wave
import struct

# Replace the content of pcm_values with your actual 16-bit PCM data
pcm_values = [
]


def pcm_to_wav(pcm_values, wav_file_path, sample_rate=192000):
    # Set up WAV file parameters
    num_channels = 1  # Mono channel
    sample_width = 2  # 16-bit PCM
    frame_rate = sample_rate  # Sampling rate (frequency)
    num_frames = len(pcm_values)
    comp_type = "NONE"
    comp_name = "not compressed"

    # Open file for writing in binary mode
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.setnframes(num_frames)
        wav_file.setcomptype(comp_type, comp_name)

        # Write the PCM data to the WAV file
        for value in pcm_values:
            # Ensure the PCM value is within the 16-bit range
            pcm_value = max(min(value, 32767), -32768)
            wav_file.writeframes(struct.pack('<h', pcm_value))


# Call the function and specify the output .wav filename
wav_file_path = 'output.wav'
pcm_to_wav(pcm_values, wav_file_path)

print(f"WAV file created at {wav_file_path}")
