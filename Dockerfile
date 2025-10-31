# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install hatch
RUN pip install --no-cache-dir hatch

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE.txt ./
COPY src/ ./src/

# Install project dependencies using hatch
RUN hatch env create

# Volume for kubeconfig
VOLUME ["/app/config/"]

# Discord token must be provided at runtime
# ENV DISCORD_TOKEN=your_token_here

# Run the application
CMD ["hatch", "run", "agent"]
