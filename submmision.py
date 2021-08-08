from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import json
import cv2
import sys
from torch import nn, Tensor
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN
import concurrent.futures

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_model() -> nn.Module:
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
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

testimages = sorted(glob('./test/images/*.jpg'))
model = get_model()
model.load_state_dict(torch.load('keypoint-rcnn-08_16_37-9.pth'))
model.eval()

with open('./sample_submission1_9.json', 'r') as f:
    data = json.load(f)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for i, image_path in tqdm(enumerate(testimages)):
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = image.transpose(2, 0, 1)
            image = [torch.as_tensor(image, dtype=torch.float32)]
            preds = model(image)
            keypoints = preds[0]['keypoints'].detach().numpy().copy()[0]
            keypoints = keypoints[:, :2]
            keypoints = keypoints.astype(np.int64)

            # print(keypoints)
            for k in range(17):
                data['annotations'][i]['joint_self'][k][0] = keypoints[k][0]
                data['annotations'][i]['joint_self'][k][1] = keypoints[k][1]
        
with open('./sample_submission1_9.json', 'w', encoding='utf-8') as mk_f:
    json.dump(data, mk_f, indent='\t', cls = NpEncoder)
# sys.exit()