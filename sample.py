
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
from flask import Flask, Response
from PIL import ImageFont, ImageDraw, Image


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
    # video_capture = cv2.VideoCapture('rtsp://192.168.0.37:8554/ds-test')
    video_capture = cv2.VideoCapture('rtsp://192.168.0.64:555')
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
    # DIVX ?????? ?????? # ?????? ?????? # DIVX, XVID, MJPG, X264, WMV1, WMV2
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # fps = 20
    # out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))
    
    p_name = ''
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    while True:
        ret, frame = video_capture.read()
        if frame is None:
            print('Image load failed')
            sys.exit()
        frame = cv2.resize(frame, (streaming_window_width ,streaming_window_height  ))
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

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
        ret, jpeg = cv2.imencode('.jpg',frame, encode_param)
        final = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n\r\n')
    video_capture.release()
    # out.release()
    cv2.destroyAllWindows()
    #writer.close()

if __name__ == "__main__":
    
    app = Flask(__name__)
    @app.route('/video')
    def video():
        return Response(writeVideo(),mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0',port=9876,threaded=True)

