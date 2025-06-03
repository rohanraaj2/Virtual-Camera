# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba
import face_recognition
import os
import cv2 
import numpy as np

# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    # If you want to use face recognition, load the model here
    known_face_encodings = []
    known_face_names = []
    # Load your known faces here
    pictures_dir = "Virtual-Camera\images"
    for picture in os.listdir(pictures_dir):
        if picture.endswith(".jpg") or picture.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(pictures_dir, picture))
            encoding = face_recognition.face_encodings(image)[0]
            if encoding is not None:
                known_face_encodings.append(encoding)
                known_face_names.append(picture.split('.')[0])

    
    for sequence in img_source_generator:
        # Call your custom processing methods here! (e. g. filters)
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
                    face_blur = cv2.GaussianBlur(face, (51, 51), 0)
                    sequence[top:bottom, left:right] = face_blur
            

            cv2.rectangle(sequence, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(sequence, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Example of keyboard is pressed
        # If you want to use this method then consider implementing a counter
        # that ignores for example the next five keyboard press events to
        # "prevent" double clicks due to high fps rates
        if keyboard.is_pressed('h') :
            print('h pressed')
            

        ###
        ### Histogram overlay example (without data)
        ###
        
        # Load the histogram values
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)        
        
        # Update the histogram with new data
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
        # uses the figure to create the overlay
        sequence = plot_overlay_to_image(sequence, fig)
        
        ###
        ### END Histogram overlay example
        ###

        
        # Display text example
        display_text_arr = ["Test", "abc"]
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