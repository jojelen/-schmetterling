Docker
======

## Useful commands

Run tensorflow jupyter server:
```
docker run -p 8888:8888 --gpus all --rm -v code:/tf/code -it tensorflow/tensorflow:latest-gpu-jupyter
```
