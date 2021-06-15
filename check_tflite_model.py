import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', help='tflite model file')
args = parser.parse_args()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

print(input_shape)
