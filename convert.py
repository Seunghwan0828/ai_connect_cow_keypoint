import json
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

# coco keypoint format

"""
{image[
    {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}]

annotation[
    {
    "keypoints": [x1,y1,v1,...],
    "num_keypoints": int,
    "[cloned]": ...,
}]

categories[{
    "keypoints": [str],
    "skeleton": [edge],
    "[cloned]": ...,
}]}
"""
train_annots = glob('./train/annotations/*.json')

coco_format = OrderedDict()
coco_format['image'] = list()
coco_format['annotation'] = list()
coco_format['categories'] = list()

for train_annot in tqdm(train_annots):
    with open(train_annot, 'r') as f:
        data = json.load(f)

        file_name = data['label_info']['image']['file_name']
        id = int(data['label_info']['image']['file_name'][-10:-4])
        width = data['label_info']['image']['width']
        height = data['label_info']['image']['height']
        bbox = data['label_info']['annotations'][0]['bbox']
        keypoints = data['label_info']['annotations'][0]['keypoints']
        keypoints_name = data['label_info']['categories'][0]['keypoints_name']
        skeleton = data['label_info']['categories'][0]['skeleton']

        coco_format['image'].append({"id": id, "width": width, "height": height, "file_name": file_name})
        coco_format['annotation'].append({"keypoints": keypoints, "num_keypoints": 17, "id": id, "image_id": id, "category_id": 1, "area": width*height, "bbox": bbox, "iscrowd": 0})
        coco_format['categories'].append({"keypoints": keypoints_name, "skeleton": skeleton})

with open('train_data.json', 'w', encoding="utf-8") as make_file:
    json.dump(coco_format, make_file, ensure_ascii=False)