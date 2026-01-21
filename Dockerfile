# Use the updated version required by the server
FROM mcr.microsoft.com/playwright/python:v1.57.0-jammy

# Set working directory
WORKDIR /app

# Install FFmpeg for video stitching
RUN apt-get update && apt-get install -y ffmpeg

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port Railway uses
EXPOSE 8080

# Start the application
CMD ["python", "main.py"]
