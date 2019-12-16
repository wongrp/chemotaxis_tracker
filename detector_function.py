import cv2
import numpy as np
import scipy
import scipy.ndimage.measurements as measurements
import sys
import os
import getopt

# takes an image and outputs to a path
def detect(im, min_box_weight):

    # set paths
    # output_picture_directory, output_picture_extension = os.path.splitext(output_path)

    w = min_box_weight # abbreviate
    s = [[1,1,1], # structuring element for labeling
         [1,1,1],
         [1,1,1]]
    im = np.uint8(im)
    if len(im) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    labeled, num_features = measurements.label(im, s) # label image
    print('number of features detected: ', num_features)
    markers = measurements.find_objects(labeled) # find labeled objects, output is slice.


    bboxes = []
    for i in range(len(markers)):
        p1 = markers[i][1].start, markers[i][0].start
        p2 = markers[i][1].stop, markers[i][0].stop
        bboxes.append([p1,p2])

    # calculate the average area
    areas = []
    for i in range(len(bboxes)):
        p1 = bboxes[i][0]
        p2 = bboxes[i][1]
        area = (p2[0]-p1[0])*(p2[1]-p1[1])
        areas.append(area)
    mean_area  = np.mean(areas)

    # threshold for box area
    for i in reversed(range(len(bboxes))):
        p1 = bboxes[i][0]
        p2 = bboxes[i][1]
        area = (p2[0]-p1[0])*(p2[1]-p1[1])
        if area < min_box_weight*mean_area:
            del bboxes[i]
        # else:
        #     cv2.rectangle(im, p1, p2, (255,255,0), 2, 1)

    print('number of objects detected:' , len(bboxes))
    # cv2.imwrite(output_path, im)


    # export bounding boxes into x y width height form
    boxes_export = {}
    current_frame_boxes = {}
    for index, box in enumerate(bboxes):
        x = box[0][0]
        y = box[0][1]
        width = box[1][0]-box[0][0]
        height = box[1][1] -box[0][1]
        mid_x = (box[0][0]+box[1][0])/2
        mid_y = (box[0][1] + box[1][1])/2
        box_dict = {"x": x, "y": y, "width": width, "height": height, "mid x": mid_x, "mid y": mid_y}
        current_frame_boxes["box " + str(index)] = box_dict

    # write text file containing bounding box information
    # with open(output_picture_directory + '.txt', 'w') as f:
    #     f.write("%s\n" % boxes_export)
    return current_frame_boxes




# takes folder of images OR an array of images, writes bounding box text file + overlaid images.
def detect_frames(min_box_weight, output_directory = None, images_array = None, input_folder = None,  num_frames = None): # folder directory
    # Load images
    if input_folder == True:
        images_array = []
        for filename in os.listdir(input_folder):
            img = cv2.imread(os.path.join(input_folder,filename))
            if img is not None:
                images_array.append(img)
    counter = 0
    boxes_export = {}
    for im in images_array:
        # Detect each frame, index them.
        # output_path = output_directory + '_' + str(counter) + '.png'
        current_frame_boxes  = detect(im, min_box_weight)
        boxes_export["frame " + str(counter)] = current_frame_boxes
        if num_frames is not None:
            if counter == num_frames-1:
                break
        counter += 1

    # # write bounding box frame information
    # with open(output_picture_directory + '.txt', 'w') as f:
    #     f.write("%s\n" % boxes_export)
    return boxes_export
