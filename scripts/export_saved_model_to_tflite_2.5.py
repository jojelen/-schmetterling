"""
Converts a saved_model to tflite file

This is just a sketch and some things are hardcoded and must be modified if one
actually wants to use this.
"""
import os

import tensorflow as tf
from tensorflow import keras
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('saved_model_dir', help='Saved model directory')
parser.add_argument('--output', help='tflite model file output',
default='model.tflite')
args = parser.parse_args()

print(tf.version.VERSION)

saved_model_dir = args.saved_model_dir

# This works with tf 2.5 at least. Experienced some bugs with 2.3.1 (at least
# for mobilenet v2 ssd).
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir,
        input_shapes={"serving_default_input_tensor" : [1,1920,1080,3]})

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8

tflite_model = converter.convert()
with open(args.output, "wb") as f:
    f.write(tflite_model)

# TFLite Interpreter to check I/O shapes.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Print I/O details.
print('Input/output details:')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
for input in input_details:
    print(input['name'],": ",  input['shape'])
for output in output_details:
    print(output['name'],": ",  output['shape'])

print('All done!')
