import sys
import os
from moviepy.editor import VideoFileClip

def slice_video_frames(video_folder, interval):
    interval = float(interval)

    # Base path for screenshots is the parent directory of the video_folder
    base_screenshots_path = os.path.join(os.path.dirname(video_folder), "screenshots")

    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.mov')):
            video_path = os.path.join(video_folder, filename)
            video_clip = VideoFileClip(video_path)

            # Define the specific folder for each video file's screenshots
            # Ensure this folder exists or create it
            video_specific_folder = os.path.join(base_screenshots_path, os.path.splitext(filename)[0])
            if not os.path.exists(video_specific_folder):
                os.makedirs(video_specific_folder)

            num_screenshots = int(video_clip.duration // interval)

            for i in range(num_screenshots):
                t = i * interval
                screenshot_filename = f"{os.path.splitext(filename)[0]}-{i + 1:03d}.jpg"
                screenshot_path = os.path.join(video_specific_folder, screenshot_filename)
                video_clip.save_frame(screenshot_path, t)

            print(f"Processed {filename}: {num_screenshots} screenshots saved.")