# Autism Support System

An AI-powered emotion recognition system designed to assist individuals with autism through real-time facial and vocal emotion detection, analysis, and support.

## Features

- **Visual Emotion Recognition**: Real-time facial emotion detection using computer vision
- **Vocal Expression Interpretation**: Speech-to-text emotion analysis with supportive responses
- **Emotional Log Analytics**: Data visualization and pattern recognition for personalized insights
- **Futuristic AI Interface**: Modern, accessible design with glassmorphism and neon aesthetics

## Tech Stack

- **Backend**: Django 4.2.6
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: TensorFlow, OpenCV, SpeechRecognition
- **Styling**: Custom CSS with glassmorphism effects
- **Deployment**: Vercel, Heroku, or Railway with GitHub Actions CI/CD

## CI/CD Deployment

This project is configured for automatic deployment using GitHub Actions. When you push to the main branch, the following happens:

1. Dependencies are installed
2. Static files are collected
3. The application is deployed to your chosen platform (Vercel, Heroku, or Railway)

## Deployment

### Railway Deployment (Recommended)

To deploy this application on Railway:

1. Make sure you have the Railway CLI installed or access to the Railway dashboard
2. Link your GitHub repository to Railway
3. Set the required environment variables:
   - `DJANGO_SECRET_KEY`: A secure Django secret key
   - `DEBUG`: Set to `False` for production
   - `EMAIL_HOST_USER` and `EMAIL_HOST_PASSWORD`: For email notifications
4. The application will automatically build and deploy using the configuration in `railway.toml`

For detailed instructions, see the [RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md) file.

### Vercel Deployment

The project also includes Vercel configuration in `vercel.json` for alternative deployment. Setup

### Option 1: Vercel (Recommended)

1. Fork this repository to your GitHub account
2. Sign up at [Vercel](https://vercel.com)
3. Import your forked repository
4. Configure the following environment variables in Vercel dashboard:
   - `DJANGO_SECRET_KEY`: Generate a new secret key
   - `DEBUG`: False
   - `EMAIL_HOST_USER`: Your email for notifications
   - `EMAIL_HOST_PASSWORD`: Your email app password

### Option 2: GitHub Actions + Vercel

1. Fork this repository
2. Connect to Vercel and get your project credentials
3. Add these secrets to your GitHub repository:
   - `VERCEL_TOKEN`: Your Vercel access token
   - `VERCEL_ORG_ID`: Your Vercel organization ID
   - `VERCEL_PROJECT_ID`: Your Vercel project ID
   - `DJANGO_SECRET_KEY`: Django secret key
   - `EMAIL_HOST_USER`: Email for notifications
   - `EMAIL_HOST_PASSWORD`: Email app password

### Option 3: Heroku

1. Fork this repository
2. Install Heroku CLI and login
3. Create a new Heroku app
4. Add buildpack: `heroku buildpacks:set heroku/python`
5. Push the repository to Heroku

### Option 4: Railway

1. Fork this repository
2. Install Railway CLI and login
3. Link your project and deploy

## Environment Variables

Required environment variables for production:

- `DJANGO_SECRET_KEY`: Secret key for Django (can use default for dev)
- `DEBUG`: Set to `False` for production
- `EMAIL_HOST_USER`: Email address for notifications
- `EMAIL_HOST_PASSWORD`: App password for email
- `EMAIL_HOST`: SMTP server (defaults to smtp.gmail.com)
- `EMAIL_PORT`: SMTP port (defaults to 587)
- `EMAIL_USE_TLS`: Whether to use TLS (defaults to True)

## Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Autism-Support-System
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run migrations:
   ```bash
   python manage.py migrate
   ```

5. Start the development server:
   ```bash
   python manage.py runserver
   ```

6. Visit `http://127.0.0.1:8000/`

## Usage

- **VISUAL_EMO**: Click "INITIALIZE SCAN" to start facial emotion recognition
- **VOCAL_INT**: Click "ACTIVATE COMMS" and then the microphone icon to start voice emotion analysis
- **LOG_ANALYTICS**: View emotion analytics and download logs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.