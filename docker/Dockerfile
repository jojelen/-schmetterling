FROM tensorflow/tensorflow:2.6.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-opencv

RUN pip install opencv-python \
    numpy \
    pandas
