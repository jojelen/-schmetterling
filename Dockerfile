FROM tensorflow/tensorflow:2.3.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip

RUN pip install opencv-python \
    numpy \
    pandas
