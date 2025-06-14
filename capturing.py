# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:41 2021

@author: droes
"""

import cv2
import numpy as np
import pyvirtualcam
from image_processing import ImageProcessor

class VirtualCamera:
    """
    Virtual Camera class that captures from real camera,
    applies image processing, and outputs to virtual camera
    """
    
    def __init__(self, camera_index=0):
        """
        Initialize the virtual camera
        Args:
            camera_index: index of the physical camera to use
        """
        self.camera_index = camera_index
        self.cap = None
        self.virtual_cam = None
        self.is_running = False
        self.processor = ImageProcessor()
        self.current_filter = "edge"
        self.show_stats = False
        
        # Available filters
        self.filters = ["edge", "blur", "sharpen", "sobel", "gabor"]
        self.filter_index = 0
        
    def capture_cv_videoInput(self):
        """
        Initialize OpenCV video capture from camera
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {self.camera_index} initialized successfully")
        
    def virtual_cam_interaction(self, width=640, height=480):
        """
        Initialize virtual camera using pyvirtualcam
        """
        try:
            self.virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30)
            print(f"Virtual camera initialized: {width}x{height} @ 30fps")
        except Exception as e:
            print(f"Failed to initialize virtual camera: {e}")
            raise
    
    def capture_screen(self):
        """
        Main capture loop that processes frames and sends to virtual camera
        """
        if not self.cap or not self.virtual_cam:
            raise RuntimeError("Camera or virtual camera not initialized")
        
        self.is_running = True
        frame_count = 0
        
        print("Starting video processing...")
        print("Controls:")
        print("  'f' - Cycle through filters")
        print("  's' - Toggle statistics display")
        print("  'q' - Quit")
        print(f"Current filter: {self.current_filter}")
        
        try:
            while self.is_running:
                # Capture frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process frame with current filter
                processed_frame, stats, entropy = self.processor.process_frame(
                    frame, filter_type=self.current_filter
                )
                
                # Add statistics overlay if enabled
                if self.show_stats:
                    processed_frame = self.processor.display_statistics(processed_frame)
                
                # Convert BGR to RGB for virtual camera
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Send to virtual camera
                self.virtual_cam.send(rgb_frame)
                self.virtual_cam.sleep_until_next_frame()
                
                # Display processed frame (optional - for debugging)
                cv2.imshow('Virtual Camera Output', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    self.cycle_filter()
                elif key == ord('s'):
                    self.toggle_statistics()
                
                frame_count += 1
                
                # Print statistics every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames - Current filter: {self.current_filter}")
                    if frame_count % 500 == 0:  # Print detailed stats every 500 frames
                        print("Channel Statistics:")
                        for channel, data in stats.items():
                            print(f"  {channel.upper()}: Mean={data['mean']:.1f}, "
                                  f"Std={data['std']:.1f}, Range=[{data['min']}-{data['max']}]")
                        print(f"Entropy values: {entropy}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            self.cleanup()
    
    def cycle_filter(self):
        """
        Cycle through available filters
        """
        self.filter_index = (self.filter_index + 1) % len(self.filters)
        self.current_filter = self.filters[self.filter_index]
        print(f"Switched to filter: {self.current_filter}")
    
    def toggle_statistics(self):
        """
        Toggle statistics display on/off
        """
        self.show_stats = not self.show_stats
        print(f"Statistics display: {'ON' if self.show_stats else 'OFF'}")
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            print("Camera released")
        
        if self.virtual_cam:
            self.virtual_cam.close()
            print("Virtual camera closed")
        
        cv2.destroyAllWindows()
        print("Cleanup completed")

def main():
    """
    Main function to run the virtual camera
    """
    try:
        # Initialize virtual camera
        virtual_cam = VirtualCamera(camera_index=0)
        
        # Setup camera capture
        virtual_cam.capture_cv_videoInput()
        
        # Setup virtual camera output
        virtual_cam.virtual_cam_interaction(width=640, height=480)
        
        # Start processing
        virtual_cam.capture_screen()
        
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()
