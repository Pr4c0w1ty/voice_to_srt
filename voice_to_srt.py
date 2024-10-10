import torch
import whisper
import time

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = whisper.load_model("turbo")

# Transcribe the audio and time the process
start_time = time.time()

# Transcribe with segment-level processing
result = model.transcribe("meow.mp3",fp16=torch.cuda.is_available(),  verbose=True)  # Set verbose=True for some logging

# Function to convert seconds to SRT time format (HH:MM:SS,milliseconds)
def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

# Write the transcription result to an .srt file and track progress
total_segments = len(result['segments'])
for i, segment in enumerate(result['segments']):
    start_time_segment = time.time()
    
    start_time_srt = seconds_to_srt_time(segment['start'])
    end_time_srt = seconds_to_srt_time(segment['end'])
    text = segment['text'].strip()

    # Save to SRT format
    with open("byland.srt", "a") as srt_file:
        srt_file.write(f"{i+1}\n")  # Subtitle index
        srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")  # Timestamps
        srt_file.write(f"{text}\n\n")  # Subtitle text


# Final time logging
total_time = time.time() - start_time
print(f"Total transcription time: {total_time:.2f} seconds")
