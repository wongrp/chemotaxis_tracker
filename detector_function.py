""" A function file with all binary detection algorithms""" 
import cv2
import numpy as np
import scipy
import scipy.ndimage.measurements as measurements
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import sys
import os
import getopt
import imutils

# takes an image and outputs to a path
def detect(im, min_box_weight, min_local_max_dist):

    w = min_box_weight # abbreviate
    s = [[1,1,1], # structuring element for labeling
         [1,1,1],
         [1,1,1]]
    im = np.uint8(im)
    if len(im) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # taken from https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
    # compute exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = scipy.ndimage.distance_transform_edt(im) # compute euclidean distance map
    localMax = peak_local_max(D, indices=False, min_distance=min_local_max_dist,
    	labels=im) # find peaks in the euclidean distance map

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then apply the Watershed algorithm
    markers, num_features = measurements.label(im, s) # label image
    markers_m, num_features_m = measurements.label(localMax, s) #
    labels = watershed(-D, markers, mask=im)
    labels_m = watershed(-D, markers_m, mask=im)
    print("[INFO] {} unique segments found".format(num_features))


    # loop over the unique labels
    # labeled, num_features = measurements.label(im, s) # label image
    # print('number of features detected: ', num_features)
    markers = measurements.find_objects(labels) # find labeled objects, output is slice.
    markers_m = measurements.find_objects(labels_m)

    bboxes = []
    bboxes_m = []
    for i in range(len(markers)):
        p1 = markers[i][1].start, markers[i][0].start
        p2 = markers[i][1].stop, markers[i][0].stop
        bboxes.append([p1,p2])

    for i in range(len(markers_m)):
        p1_m = markers_m[i][1].start, markers_m[i][0].start
        p2_m = markers_m[i][1].stop, markers_m[i][0].stop
        bboxes_m.append([p1_m,p2_m])

    # calculate the average area
    areas = []
    for i in range(len(bboxes)):
        p1 = bboxes[i][0]
        p2 = bboxes[i][1]
        area = (p2[0]-p1[0])*(p2[1]-p1[1])
        areas.append(area)
    mean_area  = np.mean(areas)

    # delete small boxes based on avg box size.
    for i in reversed(range(len(bboxes))):
        p1 = bboxes[i][0]
        p2 = bboxes[i][1]
        area = (p2[0]-p1[0])*(p2[1]-p1[1])
        if area < min_box_weight*mean_area:
            del bboxes[i]

    length_boxes = len(np.copy(bboxes))
    # the following algorithm asks the user if the following detection is 'correct'.
    im_copy = 255*np.array(np.copy(im), dtype = np.uint8)
    for i in range(length_boxes):
        p1x = bboxes[i][0][0]
        p1y = bboxes[i][0][1]
        p2x = bboxes[i][1][0]
        p2y = bboxes[i][1][1]
        cv2.rectangle(im_copy, (p1x,p1y), (p2x,p2y), (255,255,255), 2, 1)
        cv2.putText(im_copy, str(i), (p1x,p1y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('window', 1000, 1000)
    cv2.imshow('window', im_copy)

    print('Press any key on the image window to continue')
    k = cv2.waitKey(0)
    # ask user input to which boxes to delete or have multiple cells.
    close_window = False
    done_deleting = False
    rerun = []
    delete_list = []
    while close_window == False:
        while done_deleting == False:
            del_num = input("Enter a box number to delete. To clear the last deletion, press d. If none, press enter. If done, press q.")
            if del_num != '' and del_num != 'q' and del_num != 'd':
                delete_list.append(np.int(del_num))
            if del_num == 'q':
                done_deleting = True # end the deleting loop
                close_window = True
                continue
            if del_num == 'd': # this lets the user correct their deletion.
                delete_list.pop()
            if del_num =='': # done deleting but still have to edit
                done_deleting = True
        if close_window == True: # the user is done deleting AND editing.
            continue # end the ENTIRE editing loop.
        box_num = input("Enter a box number that contains more than one cell. To clear last edit, press d. If done, press q: ")
        if box_num == 'q':
            close_window = True
            continue
        if del_num == 'd': # this lets the user correct their deletion.
            rerun.pop()
        else:
            print('You have selected box ', str(box_num))
            rerun.append(np.int(box_num))
    cv2.destroyWindow('window')




    # if it's multiple cells, take all points within that box add it to the list
    for i in rerun:
        p1x = bboxes[i][0][0]
        p1y = bboxes[i][0][1]
        p2x = bboxes[i][1][0]
        p2y = bboxes[i][1][1]
        for p in range(len(bboxes_m)):
             p1x_m = bboxes_m[p][0][0]
             p1y_m = bboxes_m[p][0][1]
             p2x_m = bboxes_m[p][1][0]
             p2y_m = bboxes_m[p][1][1]
             print(p1x_m, p1y_m, p2x_m, p2y_m)
             # ' within the ball park'
             if p1x_m > p1x-10 and p1y_m > p1y-10 and p2x_m < p2x+10 and p2y_m < p2y+10 and \
             (p2x_m-p1x_m)*(p2y_m-p1y_m) > min_box_weight * mean_area: # ensure it's not a small blip
                 bboxes.append(bboxes_m[p])
        #     cv2.destroyWindow('window')
        # elif (k == 113): # q is pressed
        #     break

    # the next three for loops takes care of deleting and replacing bounding boxes.
    # delete original bouding boxes that were replaced
    for i in range(len(rerun)):
            bboxes[np.max(rerun)] = 0
            rerun.remove(np.max(rerun))

    # delete original bounding boxes that were deleted
    for i in range(len(delete_list)):
            bboxes[np.max(delete_list)] = 0
            delete_list.remove(np.max(delete_list))

    # delete all boxes set to 0
    for i in reversed(range(len(bboxes))):
        if bboxes[i] == 0:
            del bboxes[i]

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
def detect_frames(min_box_weight, min_local_max_dist, output_directory = None, images_array = None, input_folder = None,  num_frames = None): # folder directory
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
        current_frame_boxes  = detect(im, min_box_weight, min_local_max_dist)
        boxes_export["frame " + str(counter)] = current_frame_boxes
        if num_frames is not None:
            if counter == num_frames-1:
                break
        counter += 1

    # # write bounding box frame information
    # with open(output_picture_directory + '.txt', 'w') as f:
    #     f.write("%s\n" % boxes_export)
    return boxes_export
