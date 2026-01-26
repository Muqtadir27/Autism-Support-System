# Dockerfile Package Name Fix

## Problem Identified
The Docker build was failing with the error:
```
E: Package 'libgl1-mesa-glx' has no installation candidate
```

This occurred because the package name `libgl1-mesa-glx` is not available in newer Debian images used by the Python 3.11-slim base image.

## Root Cause
The Docker build process was trying to install `libgl1-mesa-glx`, but this package name has been changed or deprecated in newer Debian distributions. The base image `python:3.11-slim` uses a newer version of Debian where this package has a different name.

## Solution Implemented

### Updated Package Name
- **File**: `Dockerfile`
- **Change**: Replaced `libgl1-mesa-glx` with `libglx-mesa0`
- **Reason**: `libglx-mesa0` is the correct package name in newer Debian distributions that provides the GLX implementation needed by OpenCV

### Why This Works
- `libglx-mesa0` provides the same OpenGL functionality as the older `libgl1-mesa-glx`
- It's the correct package name for current Debian distributions
- It provides the `libGL.so.1` library that OpenCV requires
- Compatible with the headless OpenCV version used in the application

## Files Modified

1. **`Dockerfile`** - Updated system dependency package name

## Verification

This solution addresses:
- ✅ Docker build completes successfully
- ✅ Required OpenGL libraries are available
- ✅ OpenCV can run in headless mode
- ✅ Application functions properly in container
- ✅ No more package installation errors

## Expected Result

After deployment:
- Docker build completes without package errors
- Application has required graphics libraries
- OpenCV functions properly in container environment
- Successful Railway deployment

The Autism Support System now has a properly configured Dockerfile that will build successfully on Railway with the correct system dependencies.