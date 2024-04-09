import sys
import os
from moviepy.editor import VideoFileClip

def slice_video_frames(video_folder, interval):
    # Ensure interval is a number
    interval = float(interval)

    # Define the path for the screenshots folder
    screenshots_folder = os.path.join(video_folder, "screenshots")
    if not os.path.exists(screenshots_folder):
        os.makedirs(screenshots_folder)

    # Iterate through all the files in the given folder
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.wav', '.mp3', '.mov')):
            # Construct the full path to the video file
            video_path = os.path.join(video_folder, filename)
            video_clip = VideoFileClip(video_path)

            # Calculate number of screenshots to take
            num_screenshots = int(video_clip.duration // interval)

            for i in range(num_screenshots):
                # Calculate the time for the current screenshot
                t = i * interval
                # Define the screenshot's filename
                screenshot_filename = f"{os.path.splitext(filename)[0]}-ss-{i + 1}.jpg"
                screenshot_path = os.path.join(screenshots_folder, screenshot_filename)
                # Save the frame
                video_clip.save_frame(screenshot_path, t)

            print(f"Processed {filename}: {num_screenshots} screenshots saved.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_video_folder> <interval_in_seconds>")
        sys.exit(1)

    video_folder_path = sys.argv[1]
    interval_seconds = sys.argv[2]

    slice_video_frames(video_folder_path, interval_seconds)
