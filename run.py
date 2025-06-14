# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import (histogram_figure_numba, get_image_statistics, apply_linear_transformation, 
                    apply_histogram_equalization, apply_edge_detection, apply_gaussian_blur, 
                    apply_sharpen_filter, apply_gabor_filter)
import face_recognition
import os
import cv2 
import numpy as np

# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    """
    Main image processing pipeline that applies various computer vision operations
    
    This function implements all required basic operations:
    - Statistical Analysis: Mean, Mode, Std Dev, Max, Min for each RGB channel
    - Transformations: Linear transformation, Histogram equalization
    - Filters: Edge detection, Blur, Sharpen, Gabor
    - Special Feature: Face detection and recognition
    - Real-time Histogram: RGB histogram overlay
    
    Keyboard Controls:
    1 - Original image (no filter)
    2 - Linear transformation (contrast/brightness adjustment)
    3 - Histogram equalization (automatic contrast enhancement)
    4 - Edge detection (Sobel filter)
    5 - Gaussian blur (smoothing)
    6 - Sharpen filter (detail enhancement)  
    7 - Gabor filter (texture analysis)
    S - Toggle statistics display on/off
    Q - Quit application
    
    Args:
        img_source_generator: Generator yielding image frames from camera
        
    Yields:
        Processed image frames with applied operations and overlays
    """
    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
    # Processing mode control variables
    current_filter = "none"  # none, linear, equalization, edge, blur, sharpen, gabor
    show_statistics = True
    frame_counter = 0  # To control keyboard input sensitivity
    
    # If you want to use face recognition, load the model here
    known_face_encodings = []
    known_face_names = []
    # Load your known faces here
    pictures_dir = "Virtual-Camera\images"
    if os.path.exists(pictures_dir):
        for picture in os.listdir(pictures_dir):
            if picture.endswith(".jpg") or picture.endswith(".png"):
                image_path = os.path.join(pictures_dir, picture)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    encoding = face_encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(picture.split('.')[0])

    
    for sequence in img_source_generator:
        frame_counter += 1
        
        # Face detection and recognition (existing special feature)
        small_frame = cv2.resize(sequence, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = []
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # Compare the face encoding with the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found, use the first match
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

        sequence = np.ascontiguousarray(sequence, dtype=np.uint8)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "Unknown":
                face = sequence[top:bottom, left:right]
                if face.size != 0:
                    face_blur = cv2.GaussianBlur(face, (51, 51), 0)
                    sequence[top:bottom, left:right] = face_blur
            
            cv2.rectangle(sequence, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(sequence, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Keyboard controls for different processing modes (with frame counter to avoid double presses)
        if frame_counter % 10 == 0:  # Check every 10 frames to reduce sensitivity
            if keyboard.is_pressed('1'):
                current_filter = "none"
                print('Filter: None (Original)')
            elif keyboard.is_pressed('2'):
                current_filter = "linear"
                print('Filter: Linear Transformation')
            elif keyboard.is_pressed('3'):
                current_filter = "equalization"
                print('Filter: Histogram Equalization')
            elif keyboard.is_pressed('4'):
                current_filter = "edge"
                print('Filter: Edge Detection')
            elif keyboard.is_pressed('5'):
                current_filter = "blur"
                print('Filter: Gaussian Blur')
            elif keyboard.is_pressed('6'):
                current_filter = "sharpen"
                print('Filter: Sharpen')
            elif keyboard.is_pressed('7'):
                current_filter = "gabor"
                print('Filter: Gabor')
            elif keyboard.is_pressed('s'):
                show_statistics = not show_statistics
                print(f'Statistics display: {"ON" if show_statistics else "OFF"}')
        
        # Apply selected image processing filter
        processed_sequence = sequence.copy()
        
        if current_filter == "linear":
            processed_sequence = apply_linear_transformation(processed_sequence, alpha=1.3, beta=20)
        elif current_filter == "equalization":
            processed_sequence = apply_histogram_equalization(processed_sequence)
        elif current_filter == "edge":
            processed_sequence = apply_edge_detection(processed_sequence)
        elif current_filter == "blur":
            processed_sequence = apply_gaussian_blur(processed_sequence, kernel_size=15)
        elif current_filter == "sharpen":
            processed_sequence = apply_sharpen_filter(processed_sequence)
        elif current_filter == "gabor":
            processed_sequence = apply_gabor_filter(processed_sequence, theta=np.pi/4, frequency=0.6)
        
        # Use processed sequence for histogram and display
        sequence = processed_sequence        
        ###
        ### Histogram overlay (with actual data from processed image)
        ###
        
        # Load the histogram values from the processed image
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)        
        
        # Update the histogram with new data
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
        # uses the figure to create the overlay
        sequence = plot_overlay_to_image(sequence, fig)
        
        ###
        ### END Histogram overlay
        ###

        
        # Display image statistics and current filter mode
        display_text_arr = [f"Filter: {current_filter.title()}"]
        
        if show_statistics:
            # Get comprehensive image statistics
            stats_text = get_image_statistics(sequence)
            display_text_arr.extend(stats_text)
        
        # Add control instructions
        display_text_arr.extend([
            "Controls:",
            "1-7: Switch filters",
            "S: Toggle statistics",
            "Q: Quit"
        ])
        
        sequence = plot_strings_to_image(sequence, display_text_arr)

        
        # Make sure to yield your processed image
        yield sequence




def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True)
            
            # or your window screen
            # vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()