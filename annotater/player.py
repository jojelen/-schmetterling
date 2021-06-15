import os
import cv2
import numpy as np
from xml.etree.ElementTree import parse, Element, ElementTree

def capFrame(frame, width, height, new_width):
    if width > new_width:
        scale_factor = width / float(new_width)
        new_height = int(height / scale_factor)
        return cv2.resize(frame, (new_width, new_height))
    else:
        return frame

class QuitAll(Exception):
    """Quit all playing of videos"""

def addBox(box,frame_annotations, label, track_id):
    outside = int(float(box.attrib['outside']))
    if outside == 1:
        return
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
        frame_annotations[frameID].append({'id': track_id,
                                       'label': label,
                                       'outside': outside,
                                       'occluded': occluded,
                                       'xtl': xtl, 'ytl': ytl,
                                       'xbr': xbr, 'ybr': ybr})
    except IndexError:
        print('WARNING: frameID={} is out of bounds (max={})'.format(frameID,
            len(frame_annotations)))

def getCoordinates(box):
    xtl = int(float(box.attrib['xtl']))
    ytl = int(float(box.attrib['ytl']))
    xbr = int(float(box.attrib['xbr']))
    ybr = int(float(box.attrib['ybr']))

    return xtl, ytl, xbr, ybr


def sameCoord(box, other_box):
    xtl, ytl, xbr, ybr = getCoordinates(box)
    new_xtl, new_ytl, new_xbr, new_ybr = getCoordinates(other_box)
    if new_xtl != xtl:
        return False
    if new_ytl != ytl:
        return False
    if new_xbr != xbr:
        return False
    if new_ybr != ybr:
        return False

    return True


def trackIsStationary(track, length = None):
    """
    Return true/false if track is of a stationary object or not.
    """
    if length is None:
        length = len(track)
    # Collect all coordinates in lists.
    all_xtl = []
    all_ytl = []
    all_xbr = []
    all_ybr = []
    index = 0
    for box in track.iter('box'):
        xtl, ytl, xbr, ybr = getCoordinates(box)
        all_xtl.append(xtl)
        all_ytl.append(ytl)
        all_xbr.append(xbr)
        all_ybr.append(ybr)
        index += 1
        if index >= length:
            break

    # Calculate standard deviation.
    std_xtl = np.std(all_xtl)
    std_ytl = np.std(all_ytl)
    std_xbr = np.std(all_xbr)
    std_ybr = np.std(all_ybr)

    # Return false if any std deviation of a coordinate is larger than 1.
    max_std = 1.0
    if std_xtl > max_std:
        return False
    if std_ytl > max_std:
        return False
    if std_xbr > max_std:
        return False
    if std_ybr > max_std:
        return False

    return True


def openXml(xml_file):
    if xml_file is None:
        return None
    with open(xml_file) as f:
        doc = parse(f)

    return doc


def getMovingIndex(track):
    for i in range(0, len(track)):
        if not trackIsStationary(track, i + 2):
            return i

    return len(track) - 1


def extractAnnotations(doc):
    if doc is None:
        return None
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
        print('Investigating track: {} {}'.format(track.attrib['label'],
            track.attrib['id']))
        if trackIsStationary(track):
            continue

        # Get index of when the object starts to move.
        move_index = getMovingIndex(track)
        print('move_index: ', move_index)

        # Keep track of how many frames a box is still.
        max_still_frames = 30
        num_frames_still = 0
        prev_box = track.find('box')
        index = -1
        for box in track.iter('box'):
            index += 1
            # Skip boxes that are stationary from the start.
            if index < move_index:
                continue

            if sameCoord(box, prev_box):
                num_frames_still += 1
            else:
                num_frames_still = 0
            prev_box = box

            if num_frames_still > max_still_frames:
                continue
            addBox(box, frame_annotations, track.attrib['label'],
                track.attrib['id'])
    for box in root.findall('box'):
        addBox(box, frame_annotations, None, None)

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
        text = '{} {}({}, occluded)'.format(annotation['label'],
                annotation['id'], annotation['mark'])
    else:
        text = '{} {}({})'.format(annotation['label'], annotation['id'], annotation['mark'])

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


def boxIsInside(box, annotation):
    """
    Returns true/false if box is inside annotation box.
    """
    # Determine the coordinates of the intersection.
    x_a= max(box['xtl'], annotation['xtl'])
    y_a = max(box['ytl'], annotation['ytl'])
    x_b = min(box['xbr'], annotation['xbr'])
    y_b = min(box['ybr'], annotation['ybr'])

    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_area = (box['xbr'] - box['xtl'] + 1) * (box['ybr'] - box['ytl'] + 1)

    min_intersection_percentage = 0.25
    if intersection_area > min_intersection_percentage * box_area:
        return True
    else:
        return False


def markAnnotations(ground_truth, annotations):
    """
    Marks correct annotations as TP and returns the number of FN, FP and TP.
    """
    # All annotations should be marked as 'FP' at this point.

    # Go through all ground_truth detections.
    numFN = 0
    for box in ground_truth:
        detected = False
        # If one is inside an annotation, mark annotation as true positive.
        for annotation in annotations:
            if annotation['label'] != 'attention':
                continue
            if boxIsInside(box, annotation):
                annotation['mark'] = 'TP'
                detected = True

        if not detected:
            numFN += 1

    # Count positives.
    numFP = 0
    numTP = 0
    for annotation in annotations:
        #print(annotation)
        if annotation['mark'] == 'TP':
            numTP += 1
        elif annotation['mark'] == 'FP':
            numFP += 1

    return numFN, numFP, numTP

def calcPR(FN, FP, TP):
    if TP is 0 and FP is 0:
        precision = 1
    else:
        precision = TP / (TP + FP)
    if TP is 0 and FN is 0:
        recall = 1
    else:
        recall = TP / (TP + FN)

    return precision, recall

def dict_to_xml(tag, d):
    '''
    Turn a simple dict of key/value pairs into XML
    '''
    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)

    return elem

def exportMeta(xml_file, tag, d):
    """
    Export dictionary to xml file
    """
    e = dict_to_xml(tag, d)

    # Check if file exists.
    if not os.path.isfile(xml_file):
        print('Creating file: ', xml_file)
        doc = ElementTree(e)
    else:
        print('Exporting results to: ', xml_file)
        doc = openXml(xml_file)
        root = doc.getroot()
        root_meta = root.find('meta')
        if root_meta is None:
            root.insert(0, e)
        else:
            # Remove old results.
            root_results = root_meta.findall('results')
            for res in root_results:
                root_meta.remove(res)

            root_meta.insert(-1, e)

    doc.write(xml_file, xml_declaration=True)


def playAnnotatedVideo(video_file, xml_file, gt_file, no_display):
    """
    Plays video

    Annotations in xml_file and gt_file are drawn in the frame.
    """
    if not os.path.isfile(video_file):
        print('Could not find \"{}\" file'.format(video_file))
        return

    # Load annotations and ground truth.
    doc_annotations = openXml(xml_file)
    annotations = extractAnnotations(doc_annotations)
    for frame_annotation in annotations:
        for annotation in frame_annotation:
            annotation['mark'] = 'FP'
    doc_gt = openXml(gt_file)
    ground_truth = extractAnnotations(doc_gt)
    for frame_gt in ground_truth:
        for gt in frame_gt:
            gt['mark'] = 'GT'

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
    totFN = 0
    totFP = 0
    totTP = 0
    frameID = 0
    while(cap.isOpened()):
        if not no_display:
            ret, frame = cap.read()
            if ret is not True:
                break
        numFN, numFP, numTP =  markAnnotations(ground_truth[frameID], annotations[frameID])
        totFN += numFN
        totFP += numFP
        totTP += numTP

        precision, recall = calcPR(totFN, totFP, totTP)
        pr_text = 'P={:.02f}, R={:.02f}'.format(precision, recall)

        if not no_display:
            if annotations is not None:
                for annotation in annotations[frameID]:
                    frame = drawAnnotation(frame, annotation, (0, 0, 255))
            if gt_file is not None:
                for gt in ground_truth[frameID]:
                    frame = drawAnnotation(frame, gt, (0, 255, 255))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            font_color = (255, 255, 255)
            font_thickness = 2
            frame = cv2.putText(frame, pr_text, (10, 100), font,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, 'FN={}'.format(totFN), (10, 200), font,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, 'FP={}'.format(totFP), (10, 300), font,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, 'TP={}'.format(totTP), (10, 400), font,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)

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
        else:
            print(pr_text)

        frameID += 1
        if frameID >= length:
            break;

    print('Final score: P={}, R={}, FN={}, FP={}, TP={}'.format(precision,
        recall, totFN, totFP, totTP))

    results = {'precision': precision, 'recall': recall, 'FN': totFN, 'FP':
            totFP, 'TP': totTP}
    exportMeta(xml_file, 'results', results)


    cap.release()
    cv2.destroyAllWindows()

    #exportResults(xml_file, precision, recall, totFN, totFP, totTP)

    if quitAll:
        raise QuitAll

