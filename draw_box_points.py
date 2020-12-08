import cv2

# input: 2d array (box_num, coordinates)
# draws boxes on the frame and labels them.
def draw_box(box, index, frame):
    # access box elements and draw
    p1 = (int(box[0]), int(box[1])) # point 1 of new box
    p2 = (int(box[2]),int(box[3])) # point 2 of box
    cv2.rectangle(frame, p1, p2, (255,255,255), 2, 1)
    cv2.putText(frame, str(index), (p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

# # input: 2d array (box_num, coordinates)
# # draws boxes on the frame and labels them.
# def draw_box_points(boxes, frame):
#     for index, new_box in enumerate(boxes):
#         # access box elements and draw
#         p1 = (int(new_box[0]), int(new_box[1])) # point 1 of new box
#         p2 = (int(new_box[2]),int(new_box[3])) # point 2 of box
#         cv2.rectangle(frame, p1, p2, (255,255,255), 2, 1)
#         cv2.putText(frame, str(index), (p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
