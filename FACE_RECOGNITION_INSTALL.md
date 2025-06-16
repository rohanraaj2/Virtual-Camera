# Face Recognition Installation Guide for Windows

## Method 1: Pre-compiled Wheels (Recommended)
Try this first:
```bash
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp313-cp313-win_amd64.whl
pip install face-recognition
```

## Method 2: Install Build Tools (if Method 1 fails)

1. **Install Visual Studio Build Tools:**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "C++ build tools" workload
   - Make sure to include "MSVC v143" and "Windows 10 SDK"

2. **Install CMake:**
   - Download from: https://cmake.org/download/
   - During installation, check "Add CMake to system PATH"
   - Verify installation: `cmake --version`

3. **Install dlib and face-recognition:**
   ```bash
   pip install dlib
   pip install face-recognition
   ```

## Method 3: Use Conda (Alternative)
If pip continues to fail:
```bash
conda install -c conda-forge dlib
pip install face-recognition
```

## Verification
Test the installation:
```python
import face_recognition
import dlib
print("Face recognition installed successfully!")
```

## Troubleshooting
- If you get "CMake is not installed" error, restart your command prompt after installing CMake
- Make sure you're using the correct Python version (3.8-3.11 recommended for better compatibility)
- For Python 3.13, pre-compiled wheels are your best option
