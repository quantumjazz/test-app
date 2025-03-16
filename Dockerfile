# Use an official Python runtime as a base image.

FROM python:3.12-slim

# Set environment variables to prevent Python from writing pyc files
# and to run stdout/stderr unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore


# Set the working directory inside the container.
WORKDIR /app

# Copy the minimal requirements file and install dependencies.
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project into the container.
COPY . /app/

# Expose port 8080, which Cloud Run (and many other hosts) expect.
EXPOSE 8080

# Start the Flask app using Gunicorn. This assumes your Flask app instance
# is defined in src/app.py as "app".
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "debug", "src.app:app"]


