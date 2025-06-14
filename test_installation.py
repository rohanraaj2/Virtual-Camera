#!/usr/bin/env python3
"""
Test script to verify face recognition installation
Run this before running the main application
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import numpy
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ opencv-python imported successfully")
    except ImportError as e:
        print(f"✗ opencv-python failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib failed: {e}")
        return False
    
    try:
        import scipy
        print("✓ scipy imported successfully")
    except ImportError as e:
        print(f"✗ scipy failed: {e}")
        return False
    
    try:
        import PIL
        print("✓ pillow imported successfully")
    except ImportError as e:
        print(f"✗ pillow failed: {e}")
        return False
    
    try:
        import keyboard
        print("✓ keyboard imported successfully")
    except ImportError as e:
        print(f"✗ keyboard failed: {e}")
        return False
    
    try:
        import pyvirtualcam
        print("✓ pyvirtualcam imported successfully")
    except ImportError as e:
        print(f"✗ pyvirtualcam failed: {e}")
        return False
    
    try:
        import dlib
        print("✓ dlib imported successfully")
    except ImportError as e:
        print(f"✗ dlib failed: {e}")
        print("  → Install dlib following FACE_RECOGNITION_INSTALL.md")
        return False
    
    try:
        import face_recognition
        print("✓ face_recognition imported successfully")
    except ImportError as e:
        print(f"✗ face_recognition failed: {e}")
        print("  → Install face-recognition following FACE_RECOGNITION_INSTALL.md")
        return False
    
    return True

def test_face_recognition():
    """Test basic face recognition functionality"""
    print("\nTesting face recognition functionality...")
    
    try:
        import face_recognition
        import numpy as np
        
        # Create a dummy image (100x100 RGB)
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Try to find faces (should return empty list for random image)
        face_locations = face_recognition.face_locations(test_image)
        print(f"✓ Face detection test completed (found {len(face_locations)} faces in random image)")
        
        return True
        
    except Exception as e:
        print(f"✗ Face recognition test failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera access successful")
            cap.release()
            return True
        else:
            print("✗ Camera access failed - check if camera is available")
            return False
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Virtual Camera Project - Installation Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test face recognition
    if not test_face_recognition():
        success = False
    
    # Test camera
    if not test_camera():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! You can run the main application.")
        print("Run: python run.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("Check FACE_RECOGNITION_INSTALL.md for help with face recognition.")
    print("=" * 50)
