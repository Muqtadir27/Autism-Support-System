# Deployment Guide

This project is configured for automated deployment using GitHub Actions. Follow these steps to deploy your Autism Support System application.

## Prerequisites

- A GitHub account
- A fork of this repository
- Access to one of the deployment platforms (Vercel, Heroku, or Railway)

## Platform-Specific Setup

### Option 1: Deploy to Vercel

1. **Sign up for Vercel**
   - Go to [vercel.com](https://vercel.com) and create an account

2. **Import your repository**
   - Click "New Project" → "Import Git Repository"
   - Select your forked repository

3. **Configure project settings**
   - Framework Preset: `Other`
   - Root Directory: `.`
   - Build Command: `bash build_files.sh`
   - Output Directory: `staticfiles_build`
   - Install Command: `pip install -r requirements.txt`

4. **Set environment variables in Vercel dashboard:**
   - `DJANGO_SECRET_KEY`: Generate a new secret key
   - `DEBUG`: `False`
   - `EMAIL_HOST_USER`: Your email for notifications
   - `EMAIL_HOST_PASSWORD`: Your email app password

### Option 2: Deploy to Heroku

1. **Sign up for Heroku**
   - Go to [heroku.com](https://heroku.com) and create an account

2. **Create a new app**
   - Click "New" → "Create new app"
   - Give your app a name and select a region

3. **Connect to GitHub**
   - Go to the "Deploy" tab
   - Click "Connect to GitHub"
   - Search for and connect your repository

4. **Enable automatic deploys**
   - Enable "Deploy Automatically" for the main branch

5. **Set environment variables in Heroku dashboard:**
   - Go to "Settings" → "Config Vars"
   - Add the following variables:
     - `DJANGO_SECRET_KEY`: Generate a new secret key
     - `DEBUG`: `False`
     - `EMAIL_HOST_USER`: Your email for notifications
     - `EMAIL_HOST_PASSWORD`: Your email app password

### Option 3: Deploy to Railway

1. **Sign up for Railway**
   - Go to [railway.app](https://railway.app) and create an account

2. **Create a new project**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your forked repository

3. **Set environment variables in Railway dashboard:**
   - Go to "Variables" → "New Variable"
   - Add the following variables:
     - `DJANGO_SECRET_KEY`: Generate a new secret key
     - `DEBUG`: `False`
     - `EMAIL_HOST_USER`: Your email for notifications
     - `EMAIL_HOST_PASSWORD`: Your email app password

## GitHub Actions Setup

If you want to use GitHub Actions for deployment, you'll need to set up repository secrets:

### For Vercel Deployment:

1. Go to your GitHub repository → Settings → Secrets and Variables → Actions
2. Add the following secrets:
   - `VERCEL_TOKEN`: Your Vercel access token (get from Vercel dashboard → Account Settings → Tokens)
   - `VERCEL_ORG_ID`: Your Vercel organization ID (get from Vercel project settings)
   - `VERCEL_PROJECT_ID`: Your Vercel project ID (get from Vercel project settings)
   - `DJANGO_SECRET_KEY`: Your Django secret key
   - `EMAIL_HOST_USER`: Your email address
   - `EMAIL_HOST_PASSWORD`: Your email app password

### For Heroku Deployment:

1. Go to your GitHub repository → Settings → Secrets and Variables → Actions
2. Add the following secrets:
   - `HEROKU_API_KEY`: Your Heroku API key (get from Heroku dashboard → Account Settings → API Key)
   - `HEROKU_APP_NAME`: Your Heroku app name
   - `HEROKU_EMAIL`: Your Heroku email address
   - `DJANGO_SECRET_KEY`: Your Django secret key
   - `EMAIL_HOST_USER`: Your email address
   - `EMAIL_HOST_PASSWORD`: Your email app password

### For Railway Deployment:

1. Go to your GitHub repository → Settings → Secrets and Variables → Actions
2. Add the following secrets:
   - `RAILWAY_TOKEN`: Your Railway token (get from Railway dashboard → Settings → Access Tokens)
   - `DJANGO_SECRET_KEY`: Your Django secret key
   - `EMAIL_HOST_USER`: Your email address
   - `EMAIL_HOST_PASSWORD`: Your email app password

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `DJANGO_SECRET_KEY` | Secret key for Django security | Random string of characters |
| `DEBUG` | Enable/disable debug mode | `False` (production) |
| `EMAIL_HOST_USER` | Email address for notifications | `your-email@gmail.com` |
| `EMAIL_HOST_PASSWORD` | App password for email | Generated app password |
| `EMAIL_HOST` | SMTP server | `smtp.gmail.com` |
| `EMAIL_PORT` | SMTP port | `587` |
| `EMAIL_USE_TLS` | Use TLS encryption | `True` |

## Generating a Django Secret Key

To generate a new Django secret key, run:
```python
python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
```

## Gmail Setup for Notifications

If using Gmail for notifications:
1. Go to Google Account settings
2. Enable 2-factor authentication
3. Generate an App Password
4. Use the App Password as `EMAIL_HOST_PASSWORD`

## Post-Deployment

After successful deployment:
1. Visit your application URL
2. Test all functionality
3. Verify that email notifications are working
4. Check that static files are loading properly

## Troubleshooting

### Common Issues:

1. **Static files not loading**: Ensure `collectstatic` ran successfully during deployment
2. **Email notifications not working**: Check email configuration and app passwords
3. **Database migrations**: First deployment might require manual migration setup
4. **SSL/HTTPS**: Some platforms handle SSL automatically, others may require additional configuration

### Logs and Monitoring:
- Vercel: Use the Vercel dashboard to view logs
- Heroku: Use `heroku logs --tail` command
- Railway: Use the Railway dashboard to view logs

## Scaling

The application is designed to scale horizontally. For high-traffic scenarios, consider:
- Upgrading to paid plan on your chosen platform
- Setting up a proper database (PostgreSQL instead of SQLite)
- Configuring CDN for static assets
- Adding caching mechanisms