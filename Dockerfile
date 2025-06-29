# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and app
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts

# Expose Streamlit port
EXPOSE 8501

# Set Streamlit entrypoint
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 