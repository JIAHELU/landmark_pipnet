import argparse
import numpy as np
import torch
from torchvision import transforms
import cv2
import os
import glob
from pfld.pfld import PFLDInference
from pfld.mynet import PFLDInference_M
from hdface.hdface import hdface_detector
from pfld.utils import plot_pose_cube
import json
import base64


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def test_img_detection():
    img_path ='/data/cv/jiahe.lu/nniefacelib/PFPLD-Dataset/test_data/imgs/1181_28_Sports_Fan_Sports_Fan_28_31_0.png'
    model_path ='/data/cv/jiahe.lu/nniefacelib/PFPLD/models/my_pt/pfldx0.25_epoch_160.pth'
    network_flag ='PFLD'

    file_name = img_path.split('/')[-1][:-4]
    det = hdface_detector(use_cuda=False)
    checkpoint = torch.load(model_path)
    #plfd_backbone = PFLDInference().cuda()
    if network_flag =='PFLD':
        plfd_backbone = PFLDInference()
    elif network_flag =='PFLDx0.25':
        plfd_backbone = PFLDInference(width_mult=0.25)
    elif network_flag =='mbv1x0.25':
        plfd_backbone = PFLDInference_M(backbone='mobilenetv1-0.25', pretrained=False)
    elif network_flag =='shufflenetv2x0.5':
        plfd_backbone = PFLDInference_M(backbone='shufflenetv2-0.5', pretrained=False)
    elif network_flag =='mobilenetv3_small':
        plfd_backbone = PFLDInference_M(backbone='mobilenetv3_small', pretrained=False)
    plfd_backbone.load_state_dict(checkpoint)
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.cuda()
    transform = transforms.Compose([transforms.ToTensor()])

    img = cv2.imread(img_path)

    height, width = img.shape[:2]
    img_det = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = det.detect_face(img_det)
    for i in range(len(result)):
        box = result[i]['box']
        cls = result[i]['cls']
        pts = result[i]['pts']
        x1, y1, x2, y2 = box
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 25))
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size_w = int(max([w, h]) * 0.9)
        size_h = int(max([w, h]) * 0.9)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size_w // 2
        x2 = x1 + size_w
        y1 = cy - int(size_h * 0.4)
        y2 = y1 + size_h

        left = 0
        top = 0
        bottom = 0
        right = 0
        if x1 < 0:
            left = -x1
        if y1 < 0:
            top = -y1
        if x2 >= width:
            right = x2 - width
        if y2 >= height:
            bottom = y2 - height

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = img[y1:y2, x1:x2]
        print(top, bottom, left, right)
        cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

        cropped = cv2.resize(cropped, (112, 112))
        cv2.imwrite(f'./test_related/pre-annotation/json_label/{file_name}_{i}.jpg', cropped)
        
        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = transform(input).unsqueeze(0).cuda()
        pose, landmarks = plfd_backbone(input)
        poses = pose.cpu().detach().numpy()[0] * 180 / np.pi
        pre_landmark = landmarks[0]
        print("LandMark:", pre_landmark.cpu().detach().numpy().reshape(-1, 2).shape)
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
        jsontext = {'shapes':[]}
        for index, (x, y) in enumerate(pre_landmark.astype(np.int32)):
            jsontext['shapes'].append({
                "label": f"{index}",
                "points": [[float(x), float(y)]],
                "group_id": "1",
                "description": None,
                "shape_type": "point",
                "flags": {},
                "mask": None
            })
            cv2.circle(cropped, (x, y), 1, (255, 255, 0), 1)
        cv2.imwrite(f'./test_related/pre-annotation/result/show_label_{i}.jpg', cropped)
        with open(f'./test_related/pre-annotation/json_label/{file_name}_{i}.jpg', 'rb') as f:
            qrcode = base64.b64encode(f.read()).decode()
        jsontext['imagePath'] = f"{file_name}_{i}.jpg"
        jsontext['imageData'] = None
        jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))
        f = open(f'./test_related/pre-annotation/json_label/{file_name}_{i}.json', 'w')
        f.write(jsondata)
        f.close()
        # pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size_w, size_h]
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
        # for (x, y) in pre_landmark.astype(np.int32):
        #     cv2.circle(img, (x1 - left + x, y1 - bottom + y), 1, (255, 255, 0), 1)
        # plot_pose_cube(img, poses[0], poses[1], poses[2], tdx=pts['nose'][0], tdy=pts['nose'][1],
        #                size=(x2 - x1) // 2)
    # cv2.imwrite('show_label.jpg', img)
    # cv2.imshow('0', img)
    cv2.waitKey(0)


def test_img(dir_path):
    model_path ='/data/cv/jiahe.lu/nniefacelib/PFPLD/models/checkpoint/new_model_98/checkpoint_epoch_80.pth'
    network_flag ='PFLD'
    checkpoint = torch.load(model_path)
    #plfd_backbone = PFLDInference().cuda()
    if network_flag =='PFLD':
        plfd_backbone = PFLDInference()
    elif network_flag =='PFLDx0.25':
        plfd_backbone = PFLDInference(width_mult=0.25)
    elif network_flag =='mbv1x0.25':
        plfd_backbone = PFLDInference_M(backbone='mobilenetv1-0.25', pretrained=False)
    elif network_flag =='shufflenetv2x0.5':
        plfd_backbone = PFLDInference_M(backbone='shufflenetv2-0.5', pretrained=False)
    elif network_flag =='mobilenetv3_small':
        plfd_backbone = PFLDInference_M(backbone='mobilenetv3_small', pretrained=False)
    plfd_backbone.load_state_dict(checkpoint)
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.cuda()
    transform = transforms.Compose([transforms.ToTensor()])
    
    # path_list = [('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/186frame_673id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/166frame_21id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/238frame_240id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/226frame_432id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/226frame_434id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/238frame_6id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/238frame_239id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/166frame_589id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/226frame_430id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/166frame_588id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/226frame_429id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/226frame_433id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/166frame_587id-0.jpg',), ('/data/cv/jiahe.lu/nniefacelib/label_dataset/test/new_label/226frame_428id-0.jpg',)]
    path_list = []
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            img_path = os.path.join(root, file_name)
            print(img_path)
            path_list.append(img_path)

    for img_path in path_list:
        # img_path = str(img_path)[2:-3]
        # print(img_path)
        file_name = img_path.split('/')[-1][:-4]
        img = cv2.imread(img_path)

        height, width = img.shape[:2]
        cv2.imwrite(f'./test_related/pre-annotation/json_label/{file_name}.jpg', img)
        
        input = cv2.resize(img, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = transform(input).unsqueeze(0).cuda()
        pose, landmarks = plfd_backbone(input)
        poses = pose.cpu().detach().numpy()[0] * 180 / np.pi
        pre_landmark = landmarks[0]
        print("LandMark:", pre_landmark.cpu().detach().numpy().reshape(-1, 2).shape)
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
        jsontext = {'shapes':[]}
        for index, (x, y) in enumerate(pre_landmark.astype(np.int32)):
            jsontext['shapes'].append({
                "label": f"{index}",
                "points": [[float(x), float(y)]],
                "group_id": "1",
                "description": None,
                "shape_type": "point",
                "flags": {},
                "mask": None
            })
            cv2.circle(img, (x, y), 1, (255, 255, 0), 1)
        # cv2.imwrite(f'./test_related/pre-annotation/result/show_label.jpg', img)
        with open(f'./test_related/pre-annotation/json_label/{file_name}.jpg', 'rb') as f:
            qrcode = base64.b64encode(f.read()).decode()
        jsontext['imagePath'] = f"{file_name}.jpg"
        jsontext['imageData'] = qrcode
        jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))
        f = open(f'./test_related/pre-annotation/json_label/{file_name}.json', 'w')
        f.write(jsondata)
        f.close()
        # cv2.waitKey(0)

def test_video_Frame(dir_path):
    model_path ='/data/cv/jiahe.lu/nniefacelib/PFPLD/models/checkpoint/new_model_98/checkpoint_epoch_80.pth'
    network_flag ='PFLD'
    det = hdface_detector(use_cuda=False)
    checkpoint = torch.load(model_path)
    if network_flag =='PFLD':
        plfd_backbone = PFLDInference()
    elif network_flag =='PFLDx0.25':
        plfd_backbone = PFLDInference(width_mult=0.25)
    elif network_flag =='mbv1x0.25':
        plfd_backbone = PFLDInference_M(backbone='mobilenetv1-0.25', pretrained=False)
    elif network_flag =='shufflenetv2x0.5':
        plfd_backbone = PFLDInference_M(backbone='shufflenetv2-0.5', pretrained=False)
    elif network_flag =='mobilenetv3_small':
        plfd_backbone = PFLDInference_M(backbone='mobilenetv3_small', pretrained=False)
    plfd_backbone.load_state_dict(checkpoint)
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.cuda()
    transform = transforms.Compose([transforms.ToTensor()])
    
    video_name = dir_path.split('/')[-1]
    if not os.path.exists(f'./test_related/pre-annotation/{video_name}'):
        os.mkdir(f'./test_related/pre-annotation/{video_name}')
        
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            img_path = os.path.join(root, file_name)
            print(img_path)
            file_name = file_name[:-4]
            img = cv2.imread(img_path)

            height, width = img.shape[:2]
            img_det = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = det.detect_face(img_det)
            for i in range(len(result)):
                box = result[i]['box']
                cls = result[i]['cls']
                pts = result[i]['pts']
                x1, y1, x2, y2 = box
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 25))
                w = x2 - x1 + 1
                h = y2 - y1 + 1

                size_w = int(max([w, h]) * 0.9)
                size_h = int(max([w, h]) * 0.9)
                cx = x1 + w // 2
                cy = y1 + h // 2
                x1 = cx - size_w // 2
                x2 = x1 + size_w
                y1 = cy - int(size_h * 0.4)
                y2 = y1 + size_h

                left = 0
                top = 0
                bottom = 0
                right = 0
                if x1 < 0:
                    left = -x1
                if y1 < 0:
                    top = -y1
                if x2 >= width:
                    right = x2 - width
                if y2 >= height:
                    bottom = y2 - height

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(width, x2)
                y2 = min(height, y2)

                cropped = img[y1:y2, x1:x2]
                print(top, bottom, left, right)
                cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

                cropped = cv2.resize(cropped, (112, 112))
                cv2.imwrite(f'./test_related/pre-annotation/{video_name}/{file_name}_{i}.jpg', cropped)
                
                input = cv2.resize(cropped, (112, 112))
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                input = transform(input).unsqueeze(0).cuda()
                pose, landmarks = plfd_backbone(input)
                poses = pose.cpu().detach().numpy()[0] * 180 / np.pi
                pre_landmark = landmarks[0]
                print("LandMark:", pre_landmark.cpu().detach().numpy().reshape(-1, 2).shape)
                pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
                jsontext = {'shapes':[]}
                for index, (x, y) in enumerate(pre_landmark.astype(np.int32)):
                    jsontext['shapes'].append({
                        "label": f"{index}",
                        "points": [[float(x), float(y)]],
                        "group_id": "1",
                        "description": None,
                        "shape_type": "point",
                        "flags": {},
                        "mask": None
                    })
                    cv2.circle(cropped, (x, y), 1, (255, 255, 0), 1)
                # cv2.imwrite(f'./test_related/pre-annotation/result/show_label_{i}.jpg', cropped)
                with open(f'./test_related/pre-annotation/{video_name}/{file_name}_{i}.jpg', 'rb') as f:
                    qrcode = base64.b64encode(f.read()).decode()
                jsontext['imagePath'] = f"{file_name}_{i}.jpg"
                jsontext['imageData'] = None
                jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))
                f = open(f'./test_related/pre-annotation/{video_name}/{file_name}_{i}.json', 'w')
                f.write(jsondata)
                f.close()


if __name__ == "__main__":
    # test_img('/data/cv/jiahe.lu/nniefacelib/PFPLD/test_related/video_frame/video')
    test_video_Frame('/data/cv/jiahe.lu/nniefacelib/PFPLD/test_related/video_frame/landmark10062469')