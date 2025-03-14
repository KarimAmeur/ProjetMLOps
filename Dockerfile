FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . /app/

# Install packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Expose port 5001 for Streamlit
EXPOSE 5001

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=5001"]
