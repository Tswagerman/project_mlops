# Dockerfile for the entire project

# Base image
FROM python:3.11-slim

# Install essential dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy project files and folders
COPY Makefile Makefile
COPY README.md README.md
COPY data/ data/
COPY docs/ docs/
COPY models/ models/
COPY notebooks/ notebooks/
COPY pyproject.toml pyproject.toml
COPY reports/ reports/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY tests/ tests/
COPY src/ src/
COPY LICENSE LICENSE

# Set the working directory
WORKDIR /

# Install Python dependencies
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir

# Specify the entrypoint command
CMD ["make", "train"]
