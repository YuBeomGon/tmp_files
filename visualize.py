import matplotlib.pyplot as plt
import cv2

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_point(img, point, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    if len(point) == 3 :
        x, y, z = point
    else :
        x, y = point
    x_min = x - 1
    x_max = x + 1
    y_min = y - 1
    y_max = y + 1

    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=BOX_COLOR, thickness=thickness)
    return img

def visualize_points(image, points):
    img = image.copy()
#     img = image.clone().detach()
    for point in (points):
#         print(bbox)
        img = visualize_point(img, point)
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(img)
    
    

def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
#     x_min, y_min, x_max, y_max = list(map(int, bbox))
#     print(bbox)
    if len(bbox) == 4:
        x_min, y_min, w, h = (bbox)
    else :
        x_min, y_min, w, h, label = (bbox)
    x_max = x_min + w
    y_max = y_min + h
#     x_min, y_min, x_max, y_max = list(map(round, bbox))

    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=BOX_COLOR, thickness=thickness)
    return img

def visualize_boxes(image, bboxes):
    img = image.copy()
#     img = image.clone().detach()
    for bbox in (bboxes):
#         print(bbox)
        img = visualize_bbox(img, bbox)
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(img)    