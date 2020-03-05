""" This run file takes a folder of bounding box information and tracks bounding
boxes based on IOU. It outputs cell trajectories on text and excel. """
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# reading files
boxes = {}
frames = range(3)
for frame in frames:
    path = 'data/ch02_level_2/bbox_text/ch02_level_2_frame_' + str(frame) + '.txt'
    with open(path, "r") as file:
        boxes['frame ' + str(frame)] = eval(file.read())

# converts dictionary data type to array of coordinates
def dict_to_coords(dict):
    coords = np.zeros((len(dict), 4))
    box_ids = []
    for num, i in enumerate(dict):
        try:
            x1 = dict[i]['x']
            y1 = dict[i]['y']
            x2 = dict[i]['x']+dict[i]['width']
            y2 = dict[i]['y']+dict[i]['height']
        except:
            x1 = dict[i][0]
            y1 = dict[i][1]
            x2 = dict[i][2]
            y2 = dict[i][3]
        coords[num,0] = x1
        coords[num,1] = y1
        coords[num,2] = x2
        coords[num,3] = y2
        box_ids.append(num) # add the box ID, which should be the same as 'box num'.
        # for example, if we are looking at 'box 1', 'num' should equal 1.
    return coords, box_ids

# arguments are two arrays.
def calculate_iou(box_prev, box):
    mat_iou = np.zeros((len(box), len(box_prev)))
    for num_prev, i in enumerate(box_prev):
        for num, j in enumerate(box):
            # i is an array of coordinates x1,y1,x2,y2
            x1 = j[0]
            y1 = j[1]
            x2 = j[2]
            y2 = j[3]

            x1_prev = i[0]
            y1_prev = i[1]
            x2_prev = i[2]
            y2_prev = i[3]

            x1_inter = max(x1, x1_prev)
            y1_inter = max(y1, y1_prev) # up is min, down is max
            x2_inter = min(x2, x2_prev)
            y2_inter = min(y2, y2_prev)
            intersection =  max(x2_inter - x1_inter, 0)*max(y2_inter-y1_inter,0)
            # if negative, means they don't intersect
            union = (x2-x1)*(y2-y1)+(x2_prev-x1_prev)*(y2_prev-y1_prev)-intersection
            iou = intersection/union
            # the columns are current boxes. rows are previous boxes.
            mat_iou[num,num_prev] = iou

        # array with -1 on max iou, 0 on all other entries.
        mat_iou_max = np.zeros(np.shape(mat_iou))
        c___ = 0
        for i in range(len(mat_iou)):
            mat_iou_max[i,np.argmax(mat_iou[i])] = -1
            if c___ <= 3:
                mat_iou_max[i,np.argmax(mat_iou[i])] = 0
                c___ +=1
    return mat_iou, mat_iou_max


def assign_ind(mat_iou):
    # index of max entry in each column vec in mat_iou
    row_ind, col_ind = linear_sum_assignment(mat_iou)
    return row_ind, col_ind

def unmatched(box_current, box_prev, row_ind, col_ind):
    unmatched_current = []
    unmatched_prev = []
    for ind, i in enumerate(box_current):
        if ind not in row_ind:
            unmatched_current.append(ind)

    for ind, i in enumerate(box_prev):
        if ind not in col_ind:
            unmatched_prev.append(ind)

    return unmatched_current, unmatched_prev

# reshuffles box coordinates from one frame using
# indices from its iou with its previous frame.
def reconstruct(box_current, row_ind, col_ind):
    box_new = box_current
    for i in range(len(row_ind)):
        # account for different number of boxes in each frame:
        try:
            box_current[int(row_ind[i])] = box_new[int(col_ind[i])]
        except:
            continue
    return box_new

# create a counter that keeps track of cell indices
def update_max_id(current_max_box_id, potential_max_box_id):
    if potential_max_box_id > current_max_box_id:
        current_max_box_id = potential_max_box_id

    return current_max_box_id

# get coordinate array from dictionary
def track_boxes(boxes):
    #### FIXME: use a dictionary to store the coordinates with the following order:
    #### frame: box num: array(x1,y1,x2,y2)
    #### every iteration, calculate iou then assign id then get unmatched indices
    #### then reconstruct. Now obtain an array with their box id's (this is different
    #### from the row/col indices. Need to obtain this when converting dict to array function).
    #### Now put them in the dictionary under 'frame num'
    #### unmatched_prev contains stuff that disappears. We don't care about those; the only
    #### information about that is that they contribute to what unique id will be assigned to
    #### the next cell that pops up
    #### Then take unmatched_current id's and use a counter to assign them a unique
    #### id. Now take the unique id and put that into the 'frame num' of the dictionary.
    box_coords = []
    box_prev = []
    max_box_id = 0
    box_coords_dict = {}
    for frame_num, i in enumerate(boxes):
        current_frame_dict = {}
        box_current, box_ids = dict_to_coords(boxes[i])
        row_ind = []
        col_ind = []

        # first frame
        if frame_num == 0:
            for box_num, i in enumerate(box_current):
                id = box_ids[box_num]
                current_frame_dict[id] = i
            max_box_id = update_max_id(max_box_id, np.amax(box_ids))

        # begin tracking on second frame
        elif frame_num > 0: # there is no "box -1"
            # calculate iou
            mat_iou, mat_iou_max = calculate_iou(box_prev, box_current)
            # assign ind
            row_ind, col_ind = assign_ind(mat_iou_max) # rows are current boxes.
            # get unmatched ind's
            unmatched_current, unmatched_prev = unmatched(box_current, box_prev, row_ind, col_ind)
            # update (reshuffle) current frame boxes
            box_current = reconstruct(box_current, row_ind, col_ind)

            # update our current maximum box id
            max_box_id = update_max_id(max_box_id, np.amax(box_ids))

            # obtained 3 arrays: reconstructed box current, and unmatched boxes
            # from current frame and previous frames
            # now reconstruct box_ids as well:
            box_ids_new = reconstruct(box_ids, row_ind, col_ind)

            for box_num, i in enumerate(box_current):
                box_id = box_ids_new[box_num]
                current_frame_dict[box_id] = box_current[box_num]

            # unmatched current contains cells that have appeared.
            # rename these cells according to the maximum id.
            for i in unmatched_current:
                old_box_id = i
                new_box_id = max_box_id + 1
                current_frame_dict[old_box_id] = current_frame_dict.pop(new_box_id)
                max_box_id += 1

        # update coordinate array and dictionary
        box_coords_dict[frame_num] = current_frame_dict
        box_coords.append(box_current)

        # update current frame boxes as previous frame boxes.
        box_prev = box_current
    return np.asarray(box_coords), box_coords_dict

def track_centers(box_coords):
    center_coords = []
    for i in range(len(box_coords)):
        centers = np.zeros((box_coords[i].shape[0], 2))
        for j in range(len(box_coords[i])):
            x = box_coords[i][j,0] + 0.5*(box_coords[i][j,2]-box_coords[i][j,0])
            y = box_coords[i][j,1] + 0.5*(box_coords[i][j,3]-box_coords[i][j,1])
            centers[j,0] = x
            centers[j,1] = y
        center_coords.append(centers)

    center_traj = []
    return center_coords, center_traj



box_coords, box_coords_dict = track_boxes(boxes)
print(box_coords_dict)
# center_coords, center_traj = track_centers(box_coords)
