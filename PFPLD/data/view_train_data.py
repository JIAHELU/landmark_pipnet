import cv2
import os
import numpy as np
import json
import base64

txt_file = open("/data/cv/jiahe.lu/nniefacelib/PFPLD-Dataset/test_data/list.txt")
lines = txt_file.readlines()
for line in lines[4000:4300]:
    line_list = line.split(' ')
    file_path = line_list[0]
    file_name = file_path.split('/')[-1][:-4]
    landmark = line_list[76 * 2 + 1: 95 * 2 + 1]
    print(f"{file_name}: {landmark}")
    jsontext = {'shapes':[]}
    print(type(float(landmark[1])))
    for index in range(0, len(landmark), 2):
        jsontext['shapes'].append({
            "label": f"{index}",
            "points": [[float(landmark[index]) * 112, float(landmark[index + 1]) * 112]],
            "group_id": "1",
            "description": None,
            "shape_type": "point",
            "flags": {},
            "mask": None
        })
    # with open(f'/data/cv/jiahe.lu/nniefacelib/label_dataset/new_label/{file_name}.jpg', 'rb') as f:
    #     qrcode = base64.b64encode(f.read()).decode()
    img = cv2.imread(f'/data/cv/jiahe.lu/nniefacelib/PFPLD-Dataset/test_data/imgs/{file_name}.png')
    print(f'/data/cv/jiahe.lu/nniefacelib/PFPLD-Dataset/test_data/imgs/{file_name}.png')
    cv2.imwrite(f'/data/cv/jiahe.lu/nniefacelib/PFPLD/test_related/train_json/{file_name}.png', img)
    jsontext['imagePath'] = f"{file_name}.png"
    jsontext['imageData'] = None
    jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))
    f = open(f'/data/cv/jiahe.lu/nniefacelib/PFPLD/test_related/train_json/{file_name}.json', 'w')
    f.write(jsondata)
    f.close()
txt_file.close()