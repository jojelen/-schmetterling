import os
import cv2
from xml.etree.ElementTree import parse

def capFrame(frame, width, height, new_width):
    if width > new_width:
        scale_factor = width / float(new_width)
        new_height = int(height / scale_factor)
        return cv2.resize(frame, (new_width, new_height))
    else:
        return frame

class QuitAll(Exception):
    """Quit all playing of videos"""

def addBox(box,frame_annotations, label):
    outside = int(float(box.attrib['outside']))
    occluded = int(float(box.attrib['occluded']))
    xtl = int(float(box.attrib['xtl']))
    ytl = int(float(box.attrib['ytl']))
    xbr = int(float(box.attrib['xbr']))
    ybr = int(float(box.attrib['ybr']))
    if label is None:
        try:
            label = box.attrib['label']
        except KeyError:
            label = "unknown"
    frameID = int(float(box.attrib['frame']))

    # Some annotations are out of bounds by 1. Or are they starting at 1?...
    try:
        frame_annotations[frameID].append({'id': 0,
                                       'label': label,
                                       'outside': outside,
                                       'occluded': occluded,
                                       'xtl': xtl, 'ytl': ytl,
                                       'xbr': xbr, 'ybr': ybr})
    except IndexError:
        print('WARNING: frameID={} is out of bounds (max={})'.format(frameID,
            len(frame_annotations)))


def loadAnnotations(xml_file):
    if xml_file is None:
        return None
    with open(xml_file) as f:
        doc = parse(f)

    root = doc.getroot()

    #print('Meta-data in xml file:')
    #for name in root.iter('name'):
    #    print('name: {}'.format(name.text))
    for size in root.iter('size'):
        num_frames = int(size.text)
    print('num frames: {}'.format(num_frames))
    for size in root.iter('original_size'):
        print('width: {}'.format(size.findtext('width')))
        print('height: {}'.format(size.findtext('height')))

    frame_annotations = [ [] for _ in range(num_frames)]

    # We allow for tracked boxes as well as untracked (no id and label inside
    # box).
    for track in root.iter('track'):
        for box in track.iter('box'):
            addBox(box, frame_annotations, track.attrib['label'])
    for box in root.findall('box'):
        addBox(box, frame_annotations, None)

    return frame_annotations

def drawAnnotation(frame, annotation, box_color):
    """
    Returns a frame with a labeled box from the annotation
    """
    if annotation['outside'] > 0:
        return frame

    if annotation['label'] == "attention":
        box_color = (0, 255, 0)

    if annotation['occluded'] > 0:
        box_color = (0.5 * box_color[0], 0.5 * box_color[1], 0.5 * box_color[2])
        text = '{} ({}, occluded)'.format(annotation['label'],annotation['id'])
    else:
        text = '{} ({})'.format(annotation['label'],annotation['id'])

    # Draw box.
    box_thickness = 2
    frame = cv2.rectangle(frame, (annotation['xtl'], annotation['ytl']),
                                 (annotation['xbr'], annotation['ybr']),
                                 box_color, box_thickness)

    # Draw label.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    frame = cv2.putText(frame, text, (annotation['xtl'], annotation['ytl']), font,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)

    return frame

def playAnnotatedVideo(video_file, xml_file, gt_file):
    if not os.path.isfile(video_file):
        print('Could not find \"{}\" file'.format(video_file))
        return

    annotations = loadAnnotations(xml_file)
    ground_truth = loadAnnotations(gt_file)
    cap = cv2.VideoCapture(video_file)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print('Video props:')
    print('\tnum frames: {}'.format(length))
    print('\tsize: {}x{}'.format(width, height))
    print('\tfps: {}'.format(fps))

    quitAll = False
    while(cap.isOpened()):
        frameID = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if ret is not True:
            break

        if annotations is not None:
            for annotation in annotations[frameID]:
                frame = drawAnnotation(frame, annotation, (0, 0, 255))
        if gt_file is not None:
            for gt in ground_truth[frameID]:
                frame = drawAnnotation(frame, gt, (0, 255, 255))

        frame = capFrame(frame, width, height, 1080)
        # OpenCV has some trouble with åäö as window names.
        cv2.imshow(str(video_file.encode('ascii',errors='ignore')), frame)

        # Press Q on keyboard to  exit
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            quitAll = True
            break
        elif key & 0xFF == ord('n'):
            break
        elif key & 0xFF == ord('p'):
            while cv2.waitKey(10) is not ord('p'):
                pass

    cap.release()
    cv2.destroyAllWindows()

    if quitAll:
        raise QuitAll

