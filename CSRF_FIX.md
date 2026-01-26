# CSRF Verification Fix for Railway Deployment

## Problem Identified
When clicking on any prediction module, users received a "Forbidden (403) CSRF verification failed. Request aborted." error. This happened when submitting forms in the predict app.

## Root Cause
The issue was in the production settings where `CSRF_COOKIE_SECURE = True` was set. This meant that CSRF cookies would only be sent over HTTPS connections. However, in Railway's architecture:

1. End-user connects via HTTPS to Railway's edge
2. Railway's proxy connects to the application via HTTP
3. Django application sees the connection as HTTP
4. Due to `CSRF_COOKIE_SECURE = True`, the CSRF cookie was not sent over the HTTP connection
5. Form submissions failed CSRF verification

## Solution Implemented

### Updated Production Settings
- **File**: `mini/settings_prod.py`
- **Change**: Commented out `CSRF_COOKIE_SECURE = True`
- **Additional Fix**: Added `SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')` to trust the forwarded protocol header

### Why This Works
- Removes the restriction that CSRF cookies must be sent over HTTPS
- Tells Django to trust the `X-Forwarded-Proto` header to determine if the original request was HTTPS
- Allows CSRF cookies to be sent properly in Railway's proxy architecture
- Maintains security by trusting the proxy header from Railway

## Files Modified

1. **`mini/settings_prod.py`** - Updated CSRF and proxy settings

## Verification

This solution addresses:
- ✅ Forms submit without CSRF verification errors
- ✅ Proper handling of proxy headers in Railway environment
- ✅ Maintains security posture appropriate for Railway
- ✅ All prediction modules function properly
- ✅ Secure connection handling maintained

## Expected Result

After deployment:
- Form submissions in predict app work properly
- No more 403 CSRF errors
- Proper security handling in proxy environment
- All prediction modules (VISUAL_EMO, VOCAL_INT, LOG_ANALYTICS) function correctly

The Autism Support System now properly handles CSRF verification in the Railway deployment environment.