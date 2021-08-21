import argparse
import logging

from models.yolov3 import YOLOv3
from utils.misc import print_versions


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-d', dest='debug', action='store_true', help='debug mode')


def train(args):
    logging.debug('Starting training...')

def main():
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                format='%(levelname)s:%(asctime)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M')
        print_versions()

    model = YOLOv3(training=True)

    model.summary()

    train(args)

if __name__ == "__main__":
    main()


