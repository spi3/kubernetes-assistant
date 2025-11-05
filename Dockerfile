# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    docker-ce-cli \
    docker-buildx-plugin \
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
