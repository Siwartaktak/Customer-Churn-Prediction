FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy the source code
COPY src/ src/
COPY models/ models/

# Initialize an empty __init__.py file if it doesn't exist
RUN mkdir -p src/api && touch src/api/__init__.py

# Expose the port for the API
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]