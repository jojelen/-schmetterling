import tensorflow as tf
import cv2
import logging

def print_versions():
    logging.info('Tensorflow {}'.format(tf.__version__))
    logging.info('OpenCV {}'.format(cv2.__version__))
