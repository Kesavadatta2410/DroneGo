import cv2
import numpy as np
from ultralytics import YOLO
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pyswarm import pso
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the path to your YOLOv8 model

# Connect to the drone
vehicle = connect('127.0.0.1:14550', wait_ready=True) 
vehicle.mode = VehicleMode("GUIDED")
while not vehicle.mode.name == 'GUIDED':
    print("Waiting for mode change to GUIDED...")
    time.sleep(1)
print("Mode changed to GUIDED")

def update_occupancy_grid(grid, detections):
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # Update the grid
        grid[center_y, center_x] = 1  
    return grid

def objective_function(x):
    current_location = np.array([vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon])
    target_location = np.array([x[0], x[1]])
    distance = np.linalg.norm(current_location - target_location)
    return distance

def navigate_to_target(vehicle, target_location):
    lat_bounds = (vehicle.location.global_relative_frame.lat - 0.01, vehicle.location.global_relative_frame.lat + 0.01)
    lon_bounds = (vehicle.location.global_relative_frame.lon - 0.01, vehicle.location.global_relative_frame.lon + 0.01)
    bounds = [lat_bounds, lon_bounds]

    optimal_location, _ = pso(objective_function, [lat_bounds[0], lon_bounds[0]], [lat_bounds[1], lon_bounds[1]], swarmsize=10, maxiter=100)

    vehicle.simple_goto(LocationGlobalRelative(optimal_location[0], optimal_location[1], vehicle.location.global_relative_frame.alt))
    time.sleep(2)  

# Main function
def main():
    grid_size = (100, 100)  
    occupancy_grid = np.zeros(grid_size, dtype=int)  

    cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.1/80")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Extract detections
        detections = results.xyxy[0].cpu().numpy() 

        # Update the occupancy grid with detections
        occupancy_grid = update_occupancy_grid(occupancy_grid, detections)

        print(occupancy_grid)

        if len(detections) > 0:
            x1, y1, x2, y2, conf, cls = detections[0]
            target_x = int((x1 + x2) / 2)
            target_y = int((y1 + y2) / 2)
            target_location = (target_x, target_y, vehicle.location.global_relative_frame.alt)

            # Navigate to the target location using PSO
            navigate_to_target(vehicle, target_location)

        # Display the frame with detections
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()