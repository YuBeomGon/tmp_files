import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from matplotlib import pyplot as plt

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


class DrawObjects(object):
    def __init__(self, topology):
        self.topology = topology
        self.font_path = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
        # self.font = ImageFont.truetype(self.font_path, 10)
        self.pog_list = []

    def __call__(self, image, object_counts, objects, normalized_peaks, p_name):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        xmin = ymin = 0
        self.pog_list.append(p_name)
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)
                    if j == 0 :
                        xmin = x
                        ymin = y

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)

#         if len(self.pog_list) > 0 :                   
#             cv2.rectangle(image, (0, 0), (200, 100), color=BOX_COLOR, thickness=2)
#         if count > 0 :
#             cv2.rectangle(image, (xmin, ymin), (xmin+200, ymin+100), color=BOX_COLOR, thickness=2)
#             ((text_width, text_height), _) = cv2.getTextSize(p_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1) 

#         pil_image = Image.fromarray(image)
#         draw = ImageDraw.Draw(pil_image)

#         if len(self.pog_list) > 0 :                   
#             self.draw_pogs(draw)

#         if count > 0 : 
#             draw.text((xmin+5,ymin+5), str(p_name)[:20], fill=(0,0,255), font=self.font)

#         image = np.array(pil_image)

        return image

    def draw_pogs(self, draw) :
        for i, pog in enumerate(self.pog_list) :
            draw.text((5, i*15+5), str(pog)[:20], fill=(0,0,255), font=self.font)
            
            
class GetKeysBoxes(object):
    
    def __init__(self, topology):
        self.topology = topology
        self.box_threshold = 10
        
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        
        K = topology.shape[0]
        count = int(object_counts[0])
        
        boxes = []
        keys = []
        
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            point_x = []
            point_y = []
            key = []
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    # cv2.circle(image, (x, y), 3, color, 2)
                    
                    point_x.append(x)
                    point_y.append(y)
                    
                    if j == 0 or j == 3 or j ==6 :
                        key.append([x, y])
                    
            if len(point_x) > 3 :
                xmin = min(point_x)
                xmax = max(point_x)
                ymin = min(point_y)
                ymax = max(point_y)
                
                if xmax-xmin > self.box_threshold and ymax-ymin > self.box_threshold :
                    # boxes.append([xmin, ymin, xmax-xmin, ymax-ymin])
                    boxes.append([xmin, ymin, xmax, ymax])
                    
                keys.append(key)

        return boxes, keys
    
def get_matching(persons, goods_point) :  
    ids = list(persons.keys())
    diff = [ [ np.sqrt((goods_point[0] - p[0])**2 + (goods_point[1] - p[1])**2)  for p in points ]for points in persons.values()]
    
    dist = [ min(d) for d in diff]
    # print(dist)
    index_i = dist.index(min(dist))
    index_j = diff[index_i].index(min(diff[index_i]))    
    nearest_p = persons[ids[index_i]][index_j]

    return ids[index_i], nearest_p 

def personId_matching(person_ids, person_keys ) :
    # a_sub_b = [x for x in a if x not in b]
    new = [x for x in person_ids if x not in person_keys]
    remove = [x for x in person_keys if x not in person_ids]
    keep = [x for x in person_ids if x not in new]
    # new = keep = -1
    # remove = []
    # if person_id in person_keys :
    #     keep = person_id
    # else :
    #     new = person_id
    # if len(person_keys) > 0 and :
    #     remove = person_keys.remove(person_id)
    
    return new, remove, keep
    
    
def visualize_bbox(img, bbox, person_id, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    x_min, y_min, x_max, y_max = map(int, bbox)
    # print(bbox)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    # ((text_width, text_height), _) = cv2.getTextSize(person_id, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    # cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=person_id,
        org=(x_min, y_min),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, person_ids):
    img = image.copy()
    for bbox, person_id in zip(bboxes, person_ids):
        img = visualize_bbox(img, bbox, str(person_id))
        
    return img
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)   
    
    
def visualize_goods(image, person_goods, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    
    image = image.copy()
    
    for ids, values in zip(person_goods.keys(), person_goods.values()) :
        if len(values) > 0 :
            x, y = values['point']
            goods = values['goods']

            cv2.circle(image, (x, y), 10, color, 2)
            for i, good in enumerate(goods) :
                cv2.putText(
                    image,
                    text=good,
                    org=(x+10 , y + 10*i),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.35, 
                    color=TEXT_COLOR, 
                    lineType=cv2.LINE_AA,
            )            

    return image    
    
