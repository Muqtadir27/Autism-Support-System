# Camera Functionality Limitation in Web Deployment

## Current Status

The **VISUAL_EMO** (Visual Emotion Recognition) feature requires direct camera access and desktop GUI capabilities. In web deployment environments like Railway, the camera window cannot be displayed in the browser because:

1. **Desktop GUI Dependency**: The current implementation uses OpenCV to create desktop windows for camera display
2. **Server Environment Limitation**: Web deployment platforms run applications in server environments without desktop GUI support
3. **Security Restrictions**: Browsers have strict security policies that prevent direct camera access from server-side applications

## How to Use VISUAL_EMO Feature

### Option 1: Local Installation (Recommended)
1. Clone the repository to your local machine
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application locally: `python manage.py runserver`
4. Connect a camera to your computer
5. Access the application at `http://127.0.0.1:8000/`
6. Click "INITIALIZE SCAN" to start the camera

### Option 2: Future Web-Based Implementation
A web-based camera interface would require:
- HTML5 MediaDevices API for browser camera access
- JavaScript for real-time video processing
- WebSockets or AJAX for communication with backend
- Significant architectural changes to the current implementation

## Current Workaround

When users click "INITIALIZE SCAN" in the deployed version:
- The system will attempt to start the camera process
- A message will be displayed explaining the limitation
- Users will be directed to run the application locally for full functionality
- The system will still attempt to initialize the camera (which may work in some containerized environments with proper configuration)

## Technical Details

The camera functionality uses:
- OpenCV for video capture and processing
- Desktop GUI windows for display
- Threading for concurrent processing
- Direct hardware access to camera devices

This architecture is designed for desktop applications and requires modification for web-based deployment.

## Future Improvements

Potential improvements could include:
1. Implementing a web-based camera interface using HTML5
2. Creating a WebSocket connection for real-time video streaming
3. Developing a progressive web app (PWA) version
4. Using WebRTC for peer-to-peer video communication

For now, the VISUAL_EMO feature is best experienced through local installation.