import cv2
import os

def read_video(synthetic_video_path):
    # Check if the file exists and its size
    if not os.path.exists(synthetic_video_path):
        print(f"Error: File not found at {synthetic_video_path}")
    else:
        print(f"File exists: {synthetic_video_path}")
        print(f"File size: {os.path.getsize(synthetic_video_path)} bytes")

    # Try opening the video with different OpenCV backends
    synthetic_video = cv2.VideoCapture(synthetic_video_path)  # Default backend

    if not synthetic_video.isOpened():
        print(f"Error: Could not open synthetic video at {synthetic_video_path}")
        # Try FFMPEG backend
        synthetic_video = cv2.VideoCapture(synthetic_video_path, cv2.CAP_FFMPEG)
        if synthetic_video.isOpened():
            print("Opened video with FFMPEG backend.")
        else:
            print("Still could not open video. Check format or codec.")
            exit()  # Exit if video cannot be opened

    # Get FPS
    fps = synthetic_video.get(cv2.CAP_PROP_FPS)

    # Read the first frame to test if OpenCV can decode it
    success, frame = synthetic_video.read()
    if not success or frame is None:
        print("Warning: First frame could not be read. The video might be empty or unsupported by OpenCV.")
        print("Possible solutions:")
        print("1. Check if the video plays in VLC.")
        print("2. Convert the video to a standard format using FFmpeg:")
        print('   ffmpeg -i "D:/uzh_project/interface/videos/synthetic_video_.mp4" -c:v libx264 -crf 23 -preset veryfast output.mp4')
        exit()  # Stop execution

    # Print frame information
    print(f"Success: {success}, Frame Type: {type(frame)}, Frame Shape: {frame.shape if frame is not None else 'None'}")

    # Process and resize frames
    synthetic_video_frames = []
    while True:
        success, frame = synthetic_video.read()

        # If no more frames, break the loop
        if not success:
            print("Warning: No more frames to read or frame is None.")
            break

        # Check if frame is valid before resizing
        if frame is None:
            print("Warning: Read a None frame, skipping...")
            continue

        # Resize frame
        try:
            synthetic_video_frames.append(frame)
        except cv2.error as e:
            print(f"OpenCV Resize Error: {e}")

    # Release the video capture
    synthetic_video.release()

    print(f"Total frames read: {len(synthetic_video_frames)}")
    return synthetic_video_frames, fps
