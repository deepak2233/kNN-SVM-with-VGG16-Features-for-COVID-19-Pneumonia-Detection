# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Command to train the model and then start FastAPI and Streamlit
CMD python3 main.py --data_dir data/ --batch_size 64 --epochs 20 && \
    uvicorn backend.api:app --host 0.0.0.0 --port 8001 & \
    streamlit run app.py --server.port 8502 --server.address 0.0.0.0
