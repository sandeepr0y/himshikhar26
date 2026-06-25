import os
import cv2


def slice_video(video_path: str, output_path: str):
    if os.path.exists(output_path):
        print(f"Skiped slicing {video_path}")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_width = frame_width // 2

    out = cv2.VideoWriter(output_path, fourcc, fps, (output_video_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Slice from the midpoint to the end to grab the RGB right side
        rgb_only_frame = frame[:, output_video_width:]

        # write the frame to output video
        out.write(rgb_only_frame)

    cap.release()
    out.release()

    print(f"Sliced video: {video_path}")