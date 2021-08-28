import cv2
import argparse
import logging

parser = argparse.ArgumentParser(description='Image viewer')
parser.add_argument('file', type=str, help='debug mode')
parser.add_argument('-d', dest='debug', action='store_true', help='debug mode')


def main(args):
    log_level = logging.INFO

    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level,
            format='%(levelname)s:%(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M')

    logging.debug('Opening \"{}\"'.format(args.file))
    img = cv2.imread(args.file, cv2.IMREAD_COLOR)

    cv2.imshow(args.file, img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)



