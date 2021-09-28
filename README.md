Piplinjen
=========

Neural networks and stuff.

## YOLOv3

[ ] Write CNN model.
[X] Enable importing of weights from darknet.
        It runs, but hasn't been checked to be correct.
[ ] Make an inference with the imported weights.
[ ] Export to tflite, run on raspberry pi.
[ ] Train model.

### Import YOLOv3 weights

Download pre-trained weights for darknet (~237mb):
```
wget https://pjreddie.com/media/files/yolov3.weights -O res/yolov3.weights
```
