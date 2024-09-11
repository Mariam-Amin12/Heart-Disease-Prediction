# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app_ml

# Copy the current directory contents into the container
COPY . /app_ml

# Install any necessary dependencies
RUN pip install -r requirments.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=index.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]