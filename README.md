<div align="center">
### 🧠 Autism Support System
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2%2B-green.svg)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)

**AI-Powered Emotion Recognition & Support Platform for Autism Spectrum Individuals**

</div>

## 🚀 Overview

The Autism Support System is an innovative AI-driven platform that leverages computer vision and machine learning to provide real-time emotion recognition and support services. Designed with empathy and precision, this system offers individuals on the autism spectrum enhanced emotional awareness and communication assistance through advanced facial expression analysis.

### 🎯 Core Capabilities
- **Real-time Emotion Detection**: Advanced facial expression analysis using deep learning models
- **Multi-modal Support**: Integrated visual, vocal, and interactive support modules
- **Privacy-First Architecture**: Local processing ensures data security and user privacy
- **Adaptive Learning**: Continuous improvement through emotion logging and analytics

---

## 🏗️ Architecture & Technology Stack

### Core Technologies
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Django 4.2+ | Web framework & API handling |
| **ML Engine** | TensorFlow 2.15 | Deep learning emotion recognition |
| **Computer Vision** | OpenCV | Face detection & image processing |
| **Frontend** | HTML5/CSS3/JS | Real-time camera interface |
| **Data Processing** | Pandas, NumPy | Analytics & logging |
| **Deployment** | Gunicorn, Whitenoise | Production serving |

### Machine Learning Pipeline
```
Input Image → Face Detection → Feature Extraction → Emotion Classification → Confidence Scoring
     ↓
Real-time Processing & Logging
```

---

## ⚡ Key Features

### 🤖 Advanced Emotion Recognition
- **7-Class Emotion Detection**: Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral
- **Confidence Scoring**: Real-time confidence percentages for accurate assessment
- **Temporal Smoothing**: Advanced stabilization to prevent flickering between emotions
- **Multi-Strategy Fallback**: DeepFace → Haar Cascade → Caffe CNN → Statistical Analysis

### 📷 Real-Time Camera Interface
- **WebRTC Integration**: Native browser camera access via `getUserMedia()`
- **Instant Processing**: 500ms polling interval for real-time feedback
- **Visual Overlay**: Dynamic emotion labels with bounding box visualization
- **Local Processing**: All analysis occurs on-device for privacy

### 🎯 Specialized Support Modules
- **Vocal Expression Interpretation**: Audio-based emotion detection
- **Interactive Games**: Emotion flashcard learning modules
- **Automated Logging**: Excel-based emotion tracking with timestamps
- **SMS Notifications**: Distress detection with automated alerts

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended for ML processing)
- **Storage**: 2GB free space for models and dependencies
- **Camera**: WebCam-enabled device for emotion detection
- **Internet**: Initial setup and model downloads

### Development Dependencies
```bash
# Essential packages
pip install Django==4.2.15
pip install tensorflow==2.15.0
pip install opencv-python-headless==4.9.0.80
pip install pandas==2.1.4
pip install deepface==0.0.79
```

---

## 🚀 Quick Start Guide

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/Muqtadir27/Autism-Support-System.git
cd Autism-Support-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate    # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download ML Models
Place the following files in `predict/Autismfiles/`:
- `deploy.prototxt.txt` - Face detection configuration
- `res10_300x300_ssd_iter_140000.caffemodel` - Face detection model
- `fer2013_mini_XCEPTION.102-0.66.hdf5` - Emotion recognition model

### 4. Configure Environment
```bash
# Create .env file
echo "DEBUG=True
SECRET_KEY=your-secret-key-here
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=your_number
RECIPIENT_PHONE_NUMBER=recipient_number" > .env
```

### 5. Launch Application
```bash
# Run development server
python manage.py runserver

# Access the application
# Local: http://127.0.0.1:8000
# Production: https://autismsupportsystem.up.railway.app
```

---

## ⚠️ Critical: Camera Functionality Requirements

### Why Camera Requires Local Execution

**Browser Security Policies**: Modern browsers enforce strict security measures:
- **Secure Context**: Camera access restricted to `https://` or `http://localhost`
- **Hardware Isolation**: Remote servers cannot access local devices
- **Privacy Protection**: Cross-origin device access blocked by design

### Enabling Camera Functionality

**Local Development Mandatory**:
1. Clone repository to local machine
2. Execute: `python manage.py runserver`
3. Access: `http://127.0.0.1:8000`
4. Grant camera permissions when prompted
5. Camera functionality activates immediately

> 🔒 **Privacy Assurance**: All processing occurs locally; no images leave your device.

---

## 🎮 Feature Usage Guide

### Emotion Detection Module
```
Navigate → Predict → START CAMERA → Position Face → Observe Results → STOP CAMERA
```

**Real-time Processing**:
- **Live Feed**: Direct camera input processing
- **Emotion Overlay**: Visual emotion labels on faces
- **Confidence Scores**: Percentage-based accuracy indicators
- **Automatic Logging**: Timestamped emotion records

### Support Features
| Feature | Access Method | Functionality |
|---------|---------------|---------------|
| **Vocal Support** | Predict → Vocal Interpretation | Audio emotion analysis |
| **Game Mode** | Predict → Flashcard Games | Interactive learning |
| **Analytics** | Dashboard → View Stats | Emotion trends & insights |
| **Export Data** | Download Button | Excel emotion logs |

---

## 🔧 Configuration Options

### Environment Variables
```env
# Django Settings
DEBUG=False
SECRET_KEY=your_production_secret_key
ALLOWED_HOSTS=autismsupportsystem.up.railway.app,127.0.0.1,localhost

# Database (PostgreSQL recommended for production)
DATABASE_URL=postgresql://user:password@host:port/database

# Twilio for SMS notifications
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
RECIPIENT_PHONE_NUMBER=+0987654321

# ML Model Paths
EMOTION_MODEL_PATH=predict/Autismfiles/fer2013_mini_XCEPTION.102-0.66.hdf5
FACE_MODEL_PATH=predict/Autismfiles/res10_300x300_ssd_iter_140000.caffemodel
```

### Performance Tuning
```python
# In settings.py for production
DEBUG = False
SECURE_SSL_REDIRECT = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
```

---

## 📊 Machine Learning Model Architecture

### Emotion Recognition Pipeline
```
Input: 64x64 Grayscale Face Image
  ↓
Preprocessing: Normalization (0-1 scale)
  ↓
Feature Extraction: Convolutional layers
  ↓
Classification: 7-class softmax output
  ↓
Output: Emotion probabilities + Confidence score
```

### Model Specifications
- **Architecture**: Xception-based CNN
- **Input**: 64x64x1 grayscale images
- **Output**: 7 emotion classes + confidence
- **Accuracy**: 85%+ on validation datasets
- **Latency**: <100ms per frame (local execution)

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Camera Access Denied** | Browser permissions | Enable camera access for http://127.0.0.1:8000 |
| **Models Not Loading** | Missing model files | Verify files in `predict/Autismfiles/` |
| **Slow Performance** | Insufficient RAM/CPU | Close other applications, upgrade hardware |
| **Remote Camera Failure** | Browser security | Use local execution only |
| **Import Errors** | Missing dependencies | Run `pip install -r requirements.txt` |

### Debug Mode Activation
```bash
# Enable detailed logging
export DEBUG=True
python manage.py runserver --verbosity=2
```

---

## 🤝 Contributing Guidelines

### Development Workflow
```bash
# 1. Fork the repository
# 2. Create feature branch
git checkout -b feature/your-amazing-feature

# 3. Make changes with comprehensive testing
# 4. Commit with clear messages
git commit -m "feat: Add amazing feature with detailed description"

# 5. Push and create pull request
git push origin feature/your-amazing-feature
```
---

## 📄 License & Attribution

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for detailed terms.

### Third-Party Libraries
- **TensorFlow**: Deep learning framework (Apache 2.0)
- **OpenCV**: Computer vision library (BSD license)
- **Django**: Web framework (3-Clause BSD)
- **Pandas**: Data analysis library (3-Clause BSD)

---

## 🆘 Support & Community

### Getting Help
- **Issues**: Report bugs via [GitHub Issues](https://github.com/Muqtadir27/Autism-Support-System/issues)
- **Email**: Contact support at abdulmuqtadir1027@gmail.com
- **Documentation**: Check the project Wiki for detailed guides

### Feedback & Suggestions
We welcome community feedback to continuously improve the Autism Support System. Please share your experiences and suggestions to enhance accessibility and functionality.

---

<div align="center">

**🔬 Research-Driven • 🤖 AI-Powered • ❤️ Community-Focused**

*Building bridges through technology for individuals on the autism spectrum*

[⭐ Star this repository if it helped you!](https://github.com/Muqtadir27/Autism-Support-System)
[🐛 Report an issue](https://github.com/Muqtadir27/Autism-Support-System/issues/new)

</div>
