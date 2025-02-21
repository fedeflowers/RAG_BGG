# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /bot
WORKDIR /bot

# Copy the current directory contents into the container at /bot
COPY bot/ .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]