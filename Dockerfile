# SupportTicketEnv — Docker image
# Builds a minimal Python 3.11 container that runs the inference script.
#
# Build:  docker build -t support-ticket-env .
# Run:    docker run --rm \
#           -e API_BASE_URL=https://api.openai.com/v1 \
#           -e MODEL_NAME=gpt-4o-mini \
#           -e HF_TOKEN=hf_... \
#           support-ticket-env

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency manifest first to leverage Docker layer caching
COPY support_ticket_env/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY support_ticket_env/ /app/support_ticket_env/

# Set Python path so the package can be imported
ENV PYTHONPATH=/app

# Default environment variable placeholders (override at runtime)
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV HF_TOKEN=""

# Run the inference script
CMD ["python", "support_ticket_env/inference.py"]
