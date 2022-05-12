FROM python:3.9
MAINTAINER THE MLEXCHANGE TEAM

# If run with docker context in main folder
COPY docker/requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    tree 

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

WORKDIR /app/work
ENV HOME /app/work
ENV PYTHONBUFFERED=1

COPY kmeans.py kmeans.py
COPY segment.py segment.py
COPY model_validation.py model_validation.py
CMD ["echo", "running kmeans"]
