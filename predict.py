import os
from ultralytics import YOLO
import cv2
import torch

VIDEOS_DIR = os.path.join('')
video_path = os.path.join(VIDEOS_DIR, 'test11.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Failed to read the video file.")
    exit(1)
H, W, _ = frame.shape

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path_out, fourcc, fps, (W, H))

model_path = os.path.join('')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(model_path).to(device)

threshold = 0.3
frames_to_process = 5  # Increased number of frames to process

for frame_count in range(total_frames):
    processed_frames = []
    detections = []

    # Process multiple frames
    for _ in range(frames_to_process):
        if frame_count + _ < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + _)
            ret, frame = cap.read()
            if ret:
                input_frame = cv2.resize(frame, (640, 640))  # Resize to square input
                if device == 'cuda':
                    input_frame = torch.from_numpy(input_frame).to(device).permute(2, 0, 1).float() / 255.0
                results = model(input_frame)[0]
                processed_frames.append(frame)
                detections.append(results)

    if not processed_frames:
        break

    # Combine detections from multiple frames
    combined_detections = []
    for result in detections:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if score > threshold:
                combined_detections.append((x1, y1, x2, y2, score, class_id))

    # Use the first frame for output (to maintain original speed)
    output_frame = processed_frames[0]

    # Draw bounding boxes on the output frame
    for x1, y1, x2, y2, score, class_id in combined_detections:
        x1, y1, x2, y2 = [int(coord * (W/640)) for coord in [x1, y1, x2, y2]]
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(output_frame, f"{detections[0].names[int(class_id)].upper()} {score:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(output_frame)
    print(f"Processed frame {frame_count} with {len(combined_detections)} detections")

    frame_count += frames_to_process

cap.release()
out.release()
cv2.destroyAllWindows()