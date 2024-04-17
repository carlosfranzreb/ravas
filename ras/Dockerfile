FROM python:3.10

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

# copy code
COPY ./stream_processing/ ./stream_processing/
RUN mkdir ./logs

# install the app
RUN pip install --upgrade pip
COPY ./requirements.txt ./requirements.txt
COPY ./setup.py ./setup.py
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install git+https://github.com/letmaik/pyvirtualcam
RUN pip install -e .

# run the app
CMD ["python", "stream_processing/main.py", "--config", "./config.yaml"]
