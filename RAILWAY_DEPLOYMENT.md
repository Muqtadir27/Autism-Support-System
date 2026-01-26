# Deploying Autism Support System on Railway

This guide will walk you through deploying the Autism Support System on Railway, a modern platform for deploying applications.

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI**: Install the Railway CLI by following the instructions at [Railway CLI Installation](https://docs.railway.app/cli/installation)
3. **Git**: Ensure Git is installed on your system

## Deployment Steps

### Option 1: Deploy via Railway Dashboard (Recommended)

1. **Prepare your repository**:
   - Ensure all changes are committed to your Git repository
   - The repository should include all the files in this project

2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app) and sign in
   - Click "New Project"
   - Select "Deploy from GitHub" (or Git provider of choice)
   - Select your repository containing this Autism Support System

3. **Configure the project**:
   - Railway will automatically detect this as a Python/Django project
   - The `railway.toml` file will be used for configuration
   - No additional configuration needed for basic deployment

4. **Set environment variables**:
   - In the Railway dashboard, go to the "Variables" tab
   - Add the following environment variables:
     ```
     DJANGO_SECRET_KEY = [Generate a secure Django secret key]
     DEBUG = False
     EMAIL_HOST_USER = [Your email for notifications]
     EMAIL_HOST_PASSWORD = [Your email password/app key]
     ```
   - For production, you may also want to set up a PostgreSQL database:
     - Go to "Provision" and add a "PostgreSQL" plugin
     - Railway will automatically populate the DATABASE_URL variable

5. **Deploy**:
   - Railway will automatically build and deploy your application
   - Monitor the build logs in the "Logs" tab
   - Once complete, your application will be available at the assigned domain

### Option 2: Deploy via Railway CLI

1. **Install Railway CLI** (if not already done):
   ```bash
   npm install -g @railway/cli
   # Or follow installation instructions at https://docs.railway.app/cli/installation
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize the project**:
   ```bash
   cd your-autism-support-system-directory
   railway init
   ```

4. **Link to your project**:
   ```bash
   railway link
   ```

5. **Set environment variables**:
   ```bash
   railway vars set DJANGO_SECRET_KEY "[Generate a secure Django secret key]"
   railway vars set DEBUG "False"
   railway vars set EMAIL_HOST_USER "[Your email for notifications]"
   railway vars set EMAIL_HOST_PASSWORD "[Your email password/app key]"
   ```

6. **Deploy**:
   ```bash
   railway deploy
   ```

## Post-Deployment Steps

1. **Run database migrations**:
   - In the Railway dashboard, go to your project
   - Go to "Settings" → "Run Command"
   - Enter: `python manage.py migrate`
   - This will set up the database tables

2. **Collect static files** (if needed):
   - Run command: `python manage.py collectstatic --noinput`

3. **Create a superuser** (optional):
   - Run command: `python manage.py createsuperuser`
   - Follow the prompts to create an admin account

## Configuration Details

### Railway Configuration (`railway.toml`)
- Uses Heroku buildpacks for Python applications
- Sets up the proper start command for the Django application
- Configures restart policies for reliability
- Sets environment variables needed for deployment

### Build Process
- Installs dependencies from `requirements.txt`
- Collects static files
- Builds the application using Gunicorn as the WSGI server
- Makes the application available on the assigned port

### Environment Variables
- `PYTHON_VERSION`: Set to "3.11" for compatibility
- `DEBUG`: Should be "False" in production
- `DJANGO_SETTINGS_MODULE`: Points to `mini.settings`
- `ALLOWED_HOSTS`: Set to "*" to accept all hosts (production should be more restrictive)
- `DATABASE_URL`: Automatically configured if PostgreSQL plugin is added

## Scaling and Maintenance

1. **Scaling**: 
   - Go to the "Settings" tab in Railway
   - Adjust the instance size and quantity as needed

2. **Monitoring**:
   - Use the "Logs" tab to monitor application activity
   - Check for errors or unusual behavior

3. **Database Migration Updates**:
   - When you update models, run migrations manually:
   - Use the "Run Command" feature to execute: `python manage.py migrate`

## Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check the logs for dependency issues
   - Ensure all required packages are in `requirements.txt`
   - Verify that the buildpack can handle all dependencies

2. **Application Crashes**:
   - Check for missing environment variables
   - Verify database connectivity
   - Review the logs for error messages

3. **Performance Issues**:
   - Consider upgrading the instance size
   - Optimize database queries if using larger datasets
   - Monitor resource usage in the Railway dashboard

### Helpful Commands:
- Check logs: `railway logs`
- Open a shell: `railway run bash`
- Run Django commands: `railway run python manage.py [command]`

## Security Considerations

1. **Secret Keys**: Never expose `DJANGO_SECRET_KEY` in code or public repositories
2. **Environment Variables**: Store all sensitive information as Railway variables
3. **Email Configuration**: Use app passwords or OAuth tokens for email services
4. **Allowed Hosts**: In production, restrict `ALLOWED_HOSTS` to your specific domains

## Updating Your Deployment

1. Push your code changes to the linked repository
2. Railway will automatically trigger a new deployment
3. Monitor the build logs to ensure successful deployment
4. Run any necessary migration commands after deployment

Your Autism Support System should now be successfully deployed on Railway and accessible via the assigned domain!