import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from pfld.pipnet import PFLDInference
from functions import *


meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface('/data/cv/jiahe.lu/nniefacelib/PFPLD/data/meanface.txt', 10)

net = PFLDInference()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

weight_file = '/data/cv/jiahe.lu/nniefacelib/PFPLD/models/checkpoint/pipnet/checkpoint_epoch_2.pth'
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])

def demo_video(video_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.9
    det_box_scale = 1.2

    net.eval()
    if video_file == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            detections, _ = detector.detect(frame, my_thresh, 1)
            for i in range(len(detections)):
                det_xmin = detections[i][2]
                det_ymin = detections[i][3]
                det_width = detections[i][4]
                det_height = detections[i][5]
                det_xmax = det_xmin + det_width - 1
                det_ymax = det_ymin + det_height - 1

                det_xmin -= int(det_width * (det_box_scale-1)/2)
                # remove a part of top area for alignment, see paper for details
                det_ymin += int(det_height * (det_box_scale-1)/2)
                det_xmax += int(det_width * (det_box_scale-1)/2)
                det_ymax += int(det_height * (det_box_scale-1)/2)
                det_xmin = max(det_xmin, 0)
                det_ymin = max(det_ymin, 0)
                det_xmax = min(det_xmax, frame_width-1)
                det_ymax = min(det_ymax, frame_height-1)
                det_width = det_xmax - det_xmin + 1
                det_height = det_ymax - det_ymin + 1
                cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
                det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
                det_crop = cv2.resize(det_crop, (input_size, input_size))
                inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
                inputs = preprocess(inputs).unsqueeze(0)
                inputs = inputs.to(device)
                lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
                lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
                tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
                lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                lms_pred = lms_pred.cpu().numpy()
                lms_pred_merge = lms_pred_merge.cpu().numpy()
                for i in range(cfg.num_lms):
                    x_pred = lms_pred_merge[i*2] * det_width
                    y_pred = lms_pred_merge[i*2+1] * det_height
                    cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
            count += 1
            #cv2.imwrite('video_out2/'+str(count)+'.jpg', frame)
            cv2.imshow('1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

demo_video(video_file, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)
