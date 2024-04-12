# Extract .wav from Videos
import os
from moviepy.editor import VideoFileClip

def extract_audio_from_videos(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.wav')):  # Real has .wav, Fake as .mp4
            video_path = os.path.join(video_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_audio_path = os.path.join(output_folder, f"{base_name}.wav")

            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le')  # Using PCM 16-bit codec

            audio_clip.close()
            video_clip.close()

            print(f"Audio extracted and saved as {output_audio_path}")