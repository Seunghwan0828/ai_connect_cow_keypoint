import json
import sys
from typing import Tuple, List, Sequence, Callable, Dict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from visualize import draw_keypoints
import torch
from torch import nn, Tensor
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN


image = cv2.imread('./test/images/ADK2021_cow_keypoint_test_0001.jpg', cv2.COLOR_BGR2RGB)
image = image / 255.0
image = image.transpose(2, 0, 1)
image = [torch.as_tensor(image, dtype=torch.float32)]

def get_model() -> nn.Module:
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=6,
        sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=12,
        sampling_ratio=2
    )

    model = KeypointRCNN(
        backbone, 
        min_size=None, max_size=384,
        num_classes=2,
        num_keypoints=17,
        box_roi_pool=roi_pooler,
        keypoint_roi_pool=keypoint_roi_pooler
    )

    return model


model = get_model()
model.load_state_dict(torch.load('keypoint-rcnn-07_20_24-9.pth'))
model.eval()
preds = model(image)
keypoints = preds[0]['keypoints'].detach().numpy().copy()[0]
image = cv2.imread('./test/images/ADK2021_cow_keypoint_test_0001.jpg', cv2.COLOR_BGR2RGB)
keypoints = keypoints[:, :2]
keypoints = keypoints.astype(np.int64)
keypoint_names = {
    0: 'fore_head',
    1: 'neck',
    2: 'fore_spine',
    3: 'fore_right_shoulder', 
    4: 'fore_right_knee', 
    5: 'fore_right_foot', 
    6: 'fore_left_shoulder',
    7: 'fore_left_knee', 
    8: 'fore_left_foot',
    9: 'rear_spine', 
    10: 'rear_right_shoulder', 
    11: 'rear_right_knee',
    12: 'rear_right_foot', 
    13: 'rear_left_shoulder',
    14: 'rear_left_knee',
    15: 'rear_left_foot',
    16: 'hip',
}

edges = [
    [1, 2],
    [2, 3],
    [3, 4],
    [3, 7],
    [3, 10],
    [4, 5],
    [5, 6],
    [7, 8],
    [8, 9],
    [10, 11],
    [10, 14],
    [11, 12],
    [12, 13],
    [14, 15],
    [15, 16],
    [10, 17]  
]

draw_keypoints(image, keypoints, edges, keypoint_names, boxes=None, dpi=None)