# Emotion Detection - Production Deployment Guide

## Problem
When deployed to production (Railway, Heroku, etc.), emotion detection returns "neutral" for all frames.

## Root Causes
1. **DeepFace downloads models on first use** - fails in containerized/restricted environments
2. **File system permissions** - can't write to cache directory
3. **Memory constraints** - models too large or timeout
4. **Network issues** - can't download models in production

## Solution Implemented

The emotion detection now uses a **3-tier fallback strategy**:

### Tier 1: DeepFace (if available)
- Uses `detector_backend='opencv'` (lightweight)
- Sets `DEEPFACE_HOME=/tmp/deepface` for production
- Falls back if imports fail, networks timeout, or memory issues occur

### Tier 2: Haar Cascade + TensorFlow (Recommended for Production)
- Uses OpenCV's built-in Haar Cascade (no downloads needed)
- Uses pre-trained FER2013 TensorFlow model
- Fast, reliable, low memory footprint
- **This is what will work in production**

### Tier 3: Caffe CNN + TensorFlow
- Fallback if Haar Cascade finds no faces
- Slower but more accurate face detection

## For Railway/Production Deployment

### 1. Ensure Models are Cached Locally
Add to `Procfile`:
```
web: python manage.py collectstatic --noinput && python -c "from predict.AutismEmoRec import get_emotion_models; get_emotion_models()" && gunicorn mini.wsgi
```

### 2. Set Environment Variables
```
DEEPFACE_HOME=/tmp/deepface
TF_CPP_MIN_LOG_LEVEL=2
```

### 3. Memory Considerations
- Haar Cascade: ~50MB
- TensorFlow model: ~100MB
- DeepFace (optional): ~300MB+

For Railway's free tier, stick with Haar Cascade + TensorFlow (Tier 2).

### 4. Testing in Production
Monitor logs for:
```
[SUCCESS] Haar Cascade + TensorFlow detected: {emotion}
```

If you see this, emotion detection is working. If you see:
```
[NEUTRAL] No face detected with sufficient confidence
```

It means no faces detected - check camera lighting/angle, not a code issue.

## Troubleshooting

### Issue: All responses are "neutral"
**Check logs:**
- Is a face being detected? (`Haar Cascade found X faces`)
- Is emotion prediction running? (`[PRED]` messages)
- Is DeepFace failing? Look for `[FALLBACK]` messages

### Issue: Timeout/Memory errors
**Solution:** DeepFace is disabled in production by default (set to use Tier 2)

### Issue: "No module named 'deepface'"
**That's OK** - the system automatically falls back to Tier 2 (Haar Cascade)

## Testing Locally Before Deploy

```bash
# Run emotion detection test
python test_emotion_detection.py

# Check logs in Django terminal for:
# [SUCCESS] Haar Cascade + TensorFlow detected: happy
```

## Production Checklist

- [ ] Models are loading correctly (check `get_emotion_models()`)
- [ ] Haar Cascade is being used as primary fallback
- [ ] No "DeepFace" ImportError crashes (falls back gracefully)
- [ ] Logs show successful emotion detections
- [ ] Camera frames are being received (check request logs)
- [ ] Emotions are being logged to emotion_log.xlsx
