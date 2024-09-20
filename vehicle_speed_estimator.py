import os
import time
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Define class list for detection
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Initialize the tracker
tracker = Tracker()

# Initialize video capture
cap = cv2.VideoCapture('highway.mp4')

# Initialize line crossing and speed detection variables
down = {}
up = {}
counter_down = []
counter_up = []

red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames if not exists
if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

count = 0

# Video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO detection on the frame
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    # Store detected cars in list
    detections = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            detections.append([x1, y1, x2, y2])

    # Update tracker
    bbox_id = tracker.update(detections)

    # Process each bounding box for speed calculation
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Downward movement detection
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = time.time()
        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = time.time() - down[id]
                if counter_down.count(id) == 0:
                    counter_down.append(id)
                    distance = 10  # meters
                    speed_ms = distance / elapsed_time
                    speed_kmh = speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(speed_kmh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Upward movement detection
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()
        if id in up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsed_time = time.time() - up[id]
                if counter_up.count(id) == 0:
                    counter_up.append(id)
                    distance = 10
                    speed_ms = distance / elapsed_time
                    speed_kmh = speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(speed_kmh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw lines and display text
    text_color = (0, 0, 0)
    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (172, red_line_y), (774, red_line_y), red_color, 2)
    cv2.line(frame, (8, blue_line_y), (927, blue_line_y), blue_color, 2)
    cv2.putText(frame, 'Going Down - ' + str(len(counter_down)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, 'Going Up - ' + str(len(counter_up)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Save frame to folder
    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    # Write frame to output video
    out.write(frame)

    # Show frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()