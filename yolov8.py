import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time

DRONE_VIEW_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 4,
    'truck': 5,
    'dog': 6,
    'chair': 7,
    'table': 8
}

RISK_LEVELS = {
    'person': 5,
    'bicycle': 3,
    'car': 4,
    'motorcycle': 4,
    'bus': 4,
    'truck': 4,
    'dog': 2,
    'chair': 1,
    'table': 1,
    'default': 1
}

RISK_COLORS = {
    5: (0, 0, 255),     # Red for highest risk
    4: (0, 165, 255),   # Orange for high risk
    3: (0, 255, 255),   # Yellow for medium risk
    2: (0, 255, 165),   # Light green for low-medium risk
    1: (0, 255, 0)      # Green for low risk
}
url='https://10.106.28.84:8080/video'
def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(url)  
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        model = YOLO('yolov8n.pt')
        if torch.cuda.is_available():
            model.to('cuda')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Starting detection... Press 'q' to quit")

    while True:
        start_time = time.time()

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            results = model(frame)

            if results and len(results) > 0:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Get class and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = result.names[class_id]

                        # Get risk level for the detected object
                        risk_level = RISK_LEVELS.get(label, RISK_LEVELS['default'])
                        color = RISK_COLORS[risk_level]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label_text = f'{label} {confidence:.2f}'
                        if risk_level >= 4:  
                            label_text += ' WARNING!'
                            
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f'FPS: {fps:.1f}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display simulated altitude
            cv2.putText(frame, 'Simulated Drone View',
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Drone View Object Detection', frame)

        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()