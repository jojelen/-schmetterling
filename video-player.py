#!python3
import argparse

import annotater.player

parser = argparse.ArgumentParser(description='Play Video With Annotations')
parser.add_argument(dest='videofile',type=str, help='Path to a video file.')
parser.add_argument('-x', dest='xml', type=str, help='Path to annotation xml file.', default=None)
parser.add_argument('-g', dest='groundtruth', type=str, help='Path to ground truth annotation xml file.', default=None)
parser.add_argument('-d', dest='debug', action='store_true', help='debug mode')
parser.add_argument('-n', dest='nodisplay', action='store_true', help='No display mode')

def main():
    args = parser.parse_args()

    if args.debug:
        print("Starting video player in debug mode")

    player.playAnnotatedVideo(args.videofile, args.xml, args.groundtruth,
            args.nodisplay)

if __name__ == "__main__":
    main()


