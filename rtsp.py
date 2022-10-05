
import cv2
import datetime
import os
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
# from datetime import datetime
import PIL
import PIL.Image
import glob
import json
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
import shutil
import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
from PIL import ImageFont, ImageDraw, Image
import time

#kafka
from confluent_kafka import Consumer
conf = {'bootstrap.servers': '192.168.0.53:9092',
        'group.id': "NajuPractice",
        'enable.auto.commit': False}
consumer = Consumer(conf)
consumer.subscribe(["transaction"])
# from draw import DrawObjects, SaveImage
#def get_topology(fname='human_pose_2.json') :
#    human_pose = json.load(f)
#    topology = trt_pose.coco.coco_category_to_topology(human_pose)
#    return topology
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
device = torch.device('cuda')
SIZE = 384
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (0, 0, 0) # White

font_path = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
font = ImageFont.truetype(font_path, 50)


class DrawObjects(object):
    def __init__(self, topology):
        self.topology = topology
        self.font_path = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
        self.font = ImageFont.truetype(self.font_path, 20)
    def __call__(self, image, object_counts, objects, normalized_peaks, p_name):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        xmin = ymin = 0
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
                    
        ((text_width, text_height), _) = cv2.getTextSize(p_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1) 
        # print(text_width, text_height)
        if text_width < 200 :
            text_width = 200
        if text_height < 100 :
            text_height  = 100

        cv2.rectangle(image, (xmin, ymin), (xmin+text_width, ymin+text_height), color=BOX_COLOR, thickness=2)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        draw.text((xmin+5, ymin+5), p_name, fill=(0,0,255), font=self.font)

        image = np.array(pil_image)

#        cv2.putText(
#            image,
#            text='하이ggk하이하이하하핳하',
#            org=(xmin, ymin + 5),
#            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#            fontScale=0.35, 
#            color=TEXT_COLOR, 
#            lineType=cv2.LINE_AA,
#        )
        
        
        return image

def load_model(fname='human_pose.json', model_name='efficientnet_b3_baseline_att') :
    with open(fname, 'r') as f :
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    model = trt_pose.models.efficientnet_b3_baseline_att(num_parts, 2 * num_links).cuda().eval()
    # model = trt_pose.models[model_name](num_parts, 2 * num_links)
    # print(model)
    # MODEL_WEIGHTS = './checkpoints/efficientnet_b3_baseline_att_384x384_epoch_50.pth'
    MODEL_WEIGHTS = 'epoch_30.pth'
    state_dict = torch.load(MODEL_WEIGHTS)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k :
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else :
            name = k # remove `module.`
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model.cuda().eval()
def preprocess(img) :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE,SIZE), interpolation=cv2.INTER_NEAREST)
    img = img / 255.
    img = (img - mean[None, None, :])/(std[None, None, :])
    return img
def writeVideo():
    file_name = 'human_pose.json'
    with open(file_name, 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    model = load_model(fname=file_name)
    
    currentTime = datetime.datetime.now()
    # RTSP setting
    # video_capture = cv2.VideoCapture('rtsp://admin:dlsxjakdlswm@192.168.0.60:554/stream1')
    # video_capture = cv2.VideoCapture('rtsp://192.168.0.64:555')
    video_capture = cv2.VideoCapture('rtsp://192.168.0.37:8554/ds-test')
    # video_capture = cv2.VideoCapture('rtsp://192.168.0.101:3002/test')
    # video_capture = cv2.VideoCapture('rtsp://192.168.0.65:556')
#    video_capture.set(3, SIZE)  # webcam widht
#    video_capture.set(4, SIZE)  # webcap height
    # check webcam width, height
    streaming_window_width = 800
    streaming_window_height = 600
    # file name using timertsp://admin:dlsxjakdlswm@192.168.0.60:554/stream1
    fileName = str(currentTime.strftime('%Y %m %d %H %M %S'))
    path = f'./video/{fileName}.avi'
    # DIVX 코덱 적용 # 코덱 종류 # DIVX, XVID, MJPG, X264, WMV1, WMV2
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # fps = 20
    # out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))
    
    p_name = ''
    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (streaming_window_width ,streaming_window_height  ))
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
##         # print(frame.shape)
        if frame is None:
            print('Image load failed')
            sys.exit()
        img = preprocess(frame)
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
        img = img.to(device)

        msg = consumer.poll(0.001)
        if msg is None:
            pass
        else:
            msg_py = json.loads(msg.value())
            # received_time = datetime.strptime(msg_py['initial_timestamp'],"%Y-%m-%d-%H:%M:%S.%f") #time_info
            shelf_num = str(msg_py['shelf_id'])
            shelf_row = int(msg_py['row'])
            col = int(msg_py['column'])
            barcode = msg_py['barcode']
            qty = msg_py['qty']
            p_name = msg_py['name']
            print(shelf_num, shelf_row, col, barcode, qty, p_name)
        cmap, paf = model(img[None, ...])
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        frame = draw_objects(frame, counts, objects, peaks, p_name)
        # video name is streaming video.
        cv2.imshow('streaming video', frame)
        #writer.write(frame)
		# out.write(frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    video_capture.release()
    # out.release()
    cv2.destroyAllWindows()
    #writer.close()
if __name__ == "__main__":
    writeVideo()
