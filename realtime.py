import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from dronekit import LocationGlobalRelative
import threading
import time
import logging
from queue import Queue

from drone_navigation import DroneNavigationSystem  # Import your main class

class TestDroneNavigationSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.drone = DroneNavigationSystem(
            connection_string='tcp:127.0.0.1:5760',  # SITL simulator address
            model_path='yolov8n.pt'
        )
        
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'drone'):
            self.drone.cleanup()

    def test_connection(self):
        """Test drone connection and initialization"""
        try:
            self.drone.connect_drone()
            self.assertIsNotNone(self.drone.vehicle)
            self.assertEqual(self.drone.vehicle.mode.name, "GUIDED")
        except Exception as e:
            self.fail(f"Connection test failed: {str(e)}")

    def test_model_initialization(self):
        """Test YOLO model loading"""
        try:
            self.drone.initialize_model()
            self.assertIsNotNone(self.drone.model)
        except Exception as e:
            self.fail(f"Model initialization failed: {str(e)}")

    def test_occupancy_grid_update(self):
        """Test occupancy grid updates"""
        # Create mock detections
        mock_detection = Mock()
        mock_detection.conf = 0.9
        mock_detection.xyxy = np.array([[100, 100, 120, 120, 0.9, 1]])
        
        frame_shape = (480, 640, 3)
        self.drone.update_occupancy_grid([mock_detection], frame_shape)
        
        # Check if grid was updated
        self.assertIsNotNone(self.drone.occupancy_grid)
        self.assertTrue(np.any(self.drone.occupancy_grid > 0))

    def test_frame_processing(self):
        """Test frame processing pipeline"""
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        results, processed_frame = self.drone.process_frame(test_frame)
        self.assertIsNotNone(processed_frame)
        self.assertEqual(processed_frame.shape, (480, 640, 3))

    def test_emergency_landing(self):
        """Test emergency landing procedure"""
        with patch('dronekit.Vehicle') as mock_vehicle:
            self.drone.vehicle = mock_vehicle
            self.drone.emergency_land()
            mock_vehicle.mode.assert_called_with("LAND")
            self.assertFalse(self.drone.is_running)

    def test_camera_thread(self):
        """Test camera capture thread"""
        self.drone.is_running = True
        camera_thread = threading.Thread(target=self.drone.camera_thread)
        camera_thread.start()
        time.sleep(1)  # Allow thread to run
        
        # Check if frames are being captured
        self.assertFalse(self.drone.frame_queue.empty())
        self.drone.is_running = False
        camera_thread.join()

    @patch('dronekit.connect')
    def test_retry_mechanism(self, mock_connect):
        """Test connection retry mechanism"""
        # Make connection fail twice then succeed
        mock_connect.side_effect = [Exception(), Exception(), Mock()]
        
        self.drone.connect_drone()
        self.assertEqual(mock_connect.call_count, 3)

def run_integration_test():
    """Run complete integration test with SITL simulator"""
    drone = DroneNavigationSystem(connection_string='tcp:127.0.0.1:5760')
    
    try:
        # Start SITL simulator (if not already running)
        import dronekit_sitl
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()
        
        # Initialize and run system
        drone.connect_drone()
        drone.initialize_model()
        
        # Test navigation to a point
        target_location = LocationGlobalRelative(-35.363261, 149.165230, 20)
        drone.navigate_to_target((target_location.lat, target_location.lon))
        
        # Run for 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            if not drone.frame_queue.empty():
                frame = drone.frame_queue.get()
                drone.process_frame(frame)
            time.sleep(0.1)
            
    finally:
        drone.cleanup()
        sitl.stop()

def test_with_real_drone():
    """Guidelines for testing with a real drone"""
    print("""
    Real Drone Testing Checklist:
    
    1. Pre-flight Checks:
       □ Battery level > 90%
       □ GPS lock acquired
       □ All sensors reporting correctly
       □ Clear testing area
       □ Weather conditions suitable
       
    2. Initial Testing:
       □ Run in simulation mode first
       □ Test hover stability
       □ Test emergency landing
       □ Verify camera feed
       
    3. Navigation Testing:
       □ Start with small movements
       □ Test obstacle detection
       □ Verify target tracking
       □ Check position accuracy
       
    4. Safety Measures:
       □ RC controller ready
       □ Safety pilot present
       □ Testing area clear
       □ Emergency procedures reviewed
    """)

if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run integration test
    print("\nRunning integration test...")
    run_integration_test()
    
    # Show real drone testing guidelines
    test_with_real_drone()