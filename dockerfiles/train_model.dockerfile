# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY project_mlops/ project_mlops/
COPY data/ data/

# Install DVC and initialize it with Google Drive remote
RUN pip install dvc && \
    dvc init --no-scm && \
    dvc remote add -d -f public-remote gdrive://1X2P4EfkFSkOlSrUg8ugygr5blUvGojsE && \
    dvc pull && \
    rm -rf .dvc/tmp && \
    rm -rf .dvc/cache && \
    rm -rf .dvc/tmp/cache && \
    apt remove -y build-essential gcc && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "project_mlops/train_model.py"]
