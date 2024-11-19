# DroneGo

1. Object Detection and Navigation with PSO (navigation_with_pso.py)
  Description
    This script integrates YOLOv8 for object detection with a Particle Swarm Optimization (PSO) algorithm for optimized drone navigation. The drone is controlled via the         dronekit library, and it uses an occupancy grid to track detected objects.

Key Features
YOLOv8 Detection: Identifies objects in the drone's camera feed.
Occupancy Grid: Updates the grid based on detected object positions.
PSO Algorithm: Optimizes the path to a target location while avoiding obstacles.
Drone Control: Connects to a drone in GUIDED mode using the DroneKit library.

How It Works
Establishes a connection to the drone.
Continuously captures video frames.
Detects objects in the frame and updates the occupancy grid.
Optimizes the drone’s path to the detected target using PSO.
Commands the drone to move to the calculated optimal position.

Dependencies
Python libraries: cv2, numpy, ultralytics, dronekit, pyswarm
YOLOv8 model (yolov8n.pt)

Usage
Ensure the drone is ready and accessible at 127.0.0.1:14550.
Place the YOLOv8 model file (yolov8n.pt) in the working directory.

Run the script:
bash
Copy code
python navigation_with_pso.py

2. Drone View Object Detection with Risk Levels (object_detection_with_risk_levels.py)
Description
This script performs real-time object detection using YOLOv8 and assigns risk levels to detected objects. It uses a webcam or an RTSP camera feed to simulate the drone's view.

Key Features
YOLOv8 Detection: Real-time detection of multiple object classes.
Risk Assessment: Assigns and displays risk levels for detected objects.
Visual Indicators: Highlights detected objects with bounding boxes and risk-based color coding.
FPS Monitoring: Displays the frames-per-second (FPS) for performance insights.

How It Works
Captures a video feed from the specified source (RTSP or webcam).
Detects objects and their associated risk levels based on pre-defined mappings.
Displays bounding boxes, labels, and warnings on the video feed.
Provides a simulated drone view for monitoring.

Dependencies
Python libraries: cv2, numpy, ultralytics, torch
YOLOv8 model (yolov8n.pt)

Usage
Replace the url variable with your RTSP camera URL or use a connected webcam.
Place the YOLOv8 model file (yolov8n.pt) in the working directory.

Run the script:
bash
Copy code
python object_detection_with_risk_levels.py
Prerequisites
Python Version: Ensure Python 3.8 or higher is installed.
Required Libraries:
Install dependencies using pip:
bash
Copy code
pip install numpy opencv-python ultralytics dronekit pyswarm torch
YOLOv8 Model:
Download the YOLOv8 model from the Ultralytics YOLOv8 repository.

Notes
The navigation_with_pso.py script requires a drone simulator (e.g., SITL) or real drone hardware for testing.
The object_detection_with_risk_levels.py script is designed for a simulated drone view and does not include navigation.
Adjust video source URLs and other parameters as needed for your environment.

Troubleshooting
YOLOv8 Errors: Ensure the model file is correctly placed and named.
Drone Connection Issues: Verify the drone IP and port for connectivity.
Performance Bottlenecks: Reduce video resolution or adjust YOLO model size for better FPS.

Acknowledgments
These scripts utilize the following open-source tools:

Ultralytics YOLOv8
DroneKit-Python
OpenCV
