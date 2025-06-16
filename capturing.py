# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:41 2021

@author: droes
"""

import cv2
import numpy as np
import pyvirtualcam
from image_processing import ImageProcessor
import sys

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
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.camera_index} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
        
    def virtual_cam_interaction(self, width=640, height=480):
        """
        Initialize virtual camera using pyvirtualcam
        """
        try:
            self.virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30)
            print(f"Virtual camera initialized: {width}x{height} @ 30fps")
            return True
        except Exception as e:
            print(f"Failed to initialize virtual camera: {e}")
            return False
    
    def capture_screen(self):
        """
        Main capture loop that processes frames and sends to virtual camera
        """
        if not self.cap or not self.virtual_cam:
            print("Error: Camera or virtual camera not properly initialized")
            return False
        
        self.is_running = True
        frame_count = 0
        
        print("Starting video processing...")
        print("Controls:")
        print("  'f' - Cycle through filters")
        print("  's' - Toggle statistics display")
        print("  'q' or ESC - Quit")
        print("  Close window - Quit")
        print(f"Current filter: {self.current_filter}")
        
        try:
            while self.is_running:
                # Check if window was closed
                if cv2.getWindowProperty('Virtual Camera Output', cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user")
                    break
                
                # Capture frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame - camera disconnected?")
                    break
                
                # Process frame with current filter
                try:
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
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    processed_frame = frame  # Use original frame as fallback
                
                # Display processed frame
                cv2.imshow('Virtual Camera Output', processed_frame)
                
                # Handle keyboard input with timeout
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    print("Quit requested by user")
                    break
                elif key == ord('f'):
                    self.cycle_filter()
                elif key == ord('s'):
                    self.toggle_statistics()
                
                frame_count += 1
                
                # Print statistics every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames - Current filter: {self.current_filter}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"Unexpected error during capture: {e}")
        finally:
            self.cleanup()
            return True
    
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
        Clean up resources properly
        """
        print("Starting cleanup...")
        self.is_running = False
        
        # Close virtual camera first
        if self.virtual_cam:
            try:
                self.virtual_cam.close()
                print("Virtual camera closed")
            except Exception as e:
                print(f"Error closing virtual camera: {e}")
            finally:
                self.virtual_cam = None
        
        # Release camera
        if self.cap:
            try:
                self.cap.release()
                print("Camera released")
            except Exception as e:
                print(f"Error releasing camera: {e}")
            finally:
                self.cap = None
        
        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            # Force window cleanup on Windows
            cv2.waitKey(1)
            print("OpenCV windows closed")
        except Exception as e:
            print(f"Error closing windows: {e}")
        
        print("Cleanup completed")

def main():
    """
    Main function to run the virtual camera
    """
    virtual_cam = None
    
    try:
        print("=== Virtual Camera with Image Processing ===")
        print("Initializing system...")
        
        # Initialize virtual camera
        virtual_cam = VirtualCamera(camera_index=0)
        
        # Setup camera capture
        if not virtual_cam.capture_cv_videoInput():
            print("Failed to initialize camera. Exiting.")
            return 1
        
        # Setup virtual camera output
        if not virtual_cam.virtual_cam_interaction(width=640, height=480):
            print("Failed to initialize virtual camera. Exiting.")
            return 1
        
        print("System initialized successfully!")
        
        # Start processing
        if not virtual_cam.capture_screen():
            print("Processing failed. Exiting.")
            return 1
            
        print("Application finished successfully.")
        return 0
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    finally:
        # Ensure cleanup happens
        if virtual_cam:
            virtual_cam.cleanup()
        
        # Force exit if needed
        print("Exiting application...")

if __name__ == "__main__":
    exit_code = main()
    # Force exit to prevent hanging
    sys.exit(exit_code)
