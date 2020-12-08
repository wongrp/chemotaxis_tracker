from iou import *
from video_capture_function import *
from draw_box_points import *
import os.path, os
import sys
import os
import getopt
import imutils

def usage():
    script = os.path.basename(__file__)
    print("\n\nUsage:  " + script + " [options] <input picture> <output picture>")
    print('''
                    Options:
                    -h --help (help)
                    <input video>
                    <input folder> (e.g. ~/input_folder)
                    <output folder> (e.g. ~/output_folder)
                    ''')
    sys.exit()

def main():

    opts, files = getopt.getopt(sys.argv[1:], "h:", [ "help"])

    if len(files) != 3:
        usage()

    # defaults:
    parameters = {}

    # loop over options:
    for option, argument in opts:
        if option in ("-h", "--help"):
            usage()


    # split path
    input_base_picture = os.path.basename(files[0])
    input_picture_name, input_picture_extension = os.path.splitext(input_base_picture)
    input_base_text = os.path.basename(files[1])
    input_text_name, input_text_extension = os.path.splitext(input_base_text)
    output_path = files[2]
    output_picture_name, output_picture_extension = os.path.splitext(files[2])

    # parameters



    # write images into a folder
    print('converting video to images...')
    start_frame = input("Enter starting frame (first frame = 0)")
    end_frame = input("Enter ending frame. If going to the end, press enter ")
    im_list = store_images(files[0])
    if end_frame == '':
        im_list = im_list[np.int(start_frame):]
    else:
        im_list = im_list[np.int(start_frame):np.int(end_frame)]
    # im_list = im_list[:int(len(im_list)/45)]

    # read detection box text files
    boxes = {}
    num_text_files = np.int(end_frame)
    frames = range(np.int(num_text_files))
    for frame in frames:
        path = files[1] + str(frame) + '.txt'
        with open(path, "r") as file:
            boxes['frame ' + str(frame)] = eval(file.read())
    # obtain tracked dictionary via iou
    box_coords, box_coords_dict = track_boxes(boxes)

    # draw boxes on images
    for frame_index, frame in enumerate(im_list):
        f = box_coords_dict[frame_index]
        for box_id in f:
            box = (f[box_id][0], f[box_id][1], f[box_id][2], f[box_id][3])
            # draw
            draw_box(box, box_id, frame)

        # write the current frame of the video
        if output_path is not None:
            print('outputting')
            cv2.imwrite(output_picture_name + '_frame_' + str(np.int(start_frame)+(frame_index)) + '.tif', frame)

    # write bounding box frame information. Each frame has its own text file.
    with open(output_picture_name + '.txt', 'w') as f:
        f.write(str(box_coords_dict))

if __name__ == "__main__":
    main()
