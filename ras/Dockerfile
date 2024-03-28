FROM python:3.10-slim

WORKDIR /app

# install system dependencies
RUN apt update
RUN apt install -y build-essential
RUN apt install -y portaudio19-dev
RUN apt install -y ffmpeg libsm6 libxext6

# set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV MEDIAPIPE_DISABLE_GPU 1

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt ./requirements.txt
COPY ./setup.py ./setup.py
RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# copy code
COPY ./stream_processing/ ./stream_processing/
COPY ./target_feats/ ./target_feats/
COPY ./onnx/ ./onnx/

# install the application
RUN pip install -e .

# run the application
CMD ["python", "stream_processing/main.py", "--config", "./config.yaml"]
