import os
import cv2
from PIL import Image
import sys
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import numpy as np
import time
import torchvision.transforms as transforms
import trt_pose.plugins


class ParseObjects(object):
    
    def __init__(self, topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=5, max_num_parts=100, max_num_objects=100):
        self.topology = topology
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        self.cmap_window = cmap_window
        self.line_integral_samples = line_integral_samples
        self.max_num_parts = max_num_parts
        self.max_num_objects = max_num_objects
    
    def __call__(self, cmap, paf):
        
        peak_counts, peaks = trt_pose.plugins.find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        normalized_peaks = trt_pose.plugins.refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        score_graph = trt_pose.plugins.paf_score_graph(paf, self.topology, peak_counts, normalized_peaks, self.line_integral_samples)
        connections = trt_pose.plugins.assignment(score_graph, self.topology, peak_counts, self.link_threshold)
        object_counts, objects =trt_pose.plugins.connect_parts(connections, self.topology, peak_counts, self.max_num_objects)
        
        return object_counts, objects, normalized_peaks

class DrawObjects(object):
    
    def __init__(self, topology):
        self.topology = topology
        
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]

        res_array = []
        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        for i in range(count):
            sub_array = []
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                sub_sub_array = []
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    sub_sub_array.append([x,y])
                    cv2.circle(image, (x, y), 3, color, 2)
                sub_array.append(sub_sub_array)
            res_array.append(sub_array)
        

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
        #print(res_array)
        file_obj.write(str(res_array))
        file_obj.write('\n')


img_path = '/home/mingun0112/torch2trt/trt_pose/tasks/human_pose/collect/'

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = image.resize((384,384))
    #image - Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:,None, None]).div_(std[:,None,None])
    return image[None,...]

def execute(image):
    data = preprocess(image)
    cmap, paf = model(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)


    image = np.array(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    draw_objects(image, counts, objects, peaks)

    return image

with open('human_pose_2.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.efficientnet_b3_baseline_att(num_parts,2*num_links).cuda()
model = torch.nn.DataParallel(model)
MODEL_WEIGHTS = '/home/mingun0112/torch2trt/trt_pose/workspace/experiments/efficientnet_b3_baseline_att_384x384/efficient_showroom/epoch_200.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229,0.224,0.225]).cuda()

parse_objects = ParseObjects(topology)
print(parse_objects)
draw_objects = DrawObjects(topology)

file_obj = open('/home/mingun0112/torch2trt/trt_pose/tasks/human_pose/new_data.txt','a')

img_list = os.listdir(img_path)
img_list = list(filter(lambda x: x[-3:]=="jpg",img_list))
temp_idx = 1
temp_len = len(img_list)
for file_name in sorted(img_list):
    file_obj.write(file_name+":")
    path = img_path + file_name
    img = Image.open(path)
    if img is None:
        continue
    result = execute(img)
    print(str(temp_idx)+"/"+str(temp_len))
    temp_idx += 1
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # result = Image.fromarray(result)
    # result.save('new_savedir/'+file_name)
file_obj.close()
