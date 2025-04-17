FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libgtk2.0-0 \
    python3-dev \
    build-essential

WORKDIR /app

# Install Python dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY . .

# Create necessary directories
RUN mkdir -p model uploads results

# Command to run the application
#CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT:-8080}", "--workers", "1", "--timeout", "120"]
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120"]
