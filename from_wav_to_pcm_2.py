'''Converts wav file to PCM data (similar to STM32)
    And zero-pads it to be a 4096 buffer (similar to STM32)'''

import wave
import struct

# Input WAV file path
input_file_path = r"Audio_Files\500Hz_IMP23ABSU_MIC.wav"

# Define the duration in seconds (21ms)
duration_seconds = 0.021

# Open the WAV file for reading
with wave.open(input_file_path, 'rb') as wav_file:
    # Get the audio parameters
    num_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()

    print(f"Channels: {num_channels}")
    print(f"Sample Width: {sample_width * 8} bits")
    print(f"Frame Rate: {frame_rate} Hz")

    # Calculate the number of frames for the first 21ms
    num_frames = int(duration_seconds * frame_rate)

    pcm_values = []
    
    # Read audio frames and collect PCM values in chunks of 4096 samples
    buffer_size = 4096
    while num_frames > 0:
        chunk_size = min(buffer_size, num_frames)
        frames = wav_file.readframes(chunk_size)
        pcm_chunk = struct.unpack(f"{len(frames) // sample_width}h", frames)
        pcm_values.extend(pcm_chunk)

        # If the chunk is smaller than buffer_size, pad with zeros
        if len(pcm_chunk) < buffer_size:
            pcm_values.extend([0] * (buffer_size - len(pcm_chunk)))

        num_frames -= chunk_size
    
    # Format PCM values as a list-like representation
    pcm_values_str = "[" + ", ".join(map(str, pcm_values)) + "]"

    print("PCM Values:")
    print(pcm_values_str)
    print(f"Number of PCM Values in First 21ms: {len(pcm_values)}")
