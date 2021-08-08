import os
from typing import Tuple, List, Sequence, Callable, Dict
from glob import glob
from collections import defaultdict
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast, grad_scaler
from datetime import datetime


class KeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_path: os.PathLike,
        transforms: Sequence[Callable]=None
    ) -> None:
        self.image_dir = image_dir
        with open(label_path, 'r') as anno_file:
            self.label_path = json.load(anno_file)

        self.image_info = self.label_path['image']
        
        self.annot_info = defaultdict(list)
        for ann in self.label_path['annotation']:
            self.annot_info[ann['image_id']].append(ann) 

        self.transforms = transforms

    def __len__(self) -> int:
        # return len(glob('./train/images/*.jpg'))
        return len(self.image_info)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_info = self.image_info[index]  # -> {}
        image_id = image_info['id']
        image_name = image_info['file_name']

        # print(image_info)
        # print(image_id)

        annots = self.annot_info.get(image_id)
        # print(annots)
        keypoints = annots[0]['keypoints']  # keypoints [[[]]]
        keypoints = np.array(keypoints).reshape(17, -1)
        for i in range(len(keypoints)):
            keypoints[i, 2] = keypoints[i,2] -1
        labels = annots[0]['category_id']  # label []
        labels = np.array(labels)
        boxes = annots[0]['bbox']  # box [[]] [x,y,wh?]
        boxes = np.array(boxes)
        # keyponts = annots['keyponts']
        # print(keypoints,'\n',labels,'\n',boxes)



        # image_name = self.label_path['image'][index]['file_name']
        # labels = np.array([1])
        # keypoints = self.label_path['annotation'][index]['keypoints']
        # boxes = self.label_path['annotation'][index]['bbox']

        image = cv2.imread(os.path.join(self.image_dir, image_name), cv2.COLOR_BGR2RGB)
        # [H, W, C]
        # [C, H, W] 0-1
        # print(image.shape)
        targets ={
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets['image']
        image = image / 255.0

        targets = {
            'labels': torch.as_tensor(targets['labels'][np.newaxis], dtype=torch.int64),
            'boxes': torch.as_tensor(targets['bboxes'][np.newaxis], dtype=torch.float32),
            'keypoints': torch.as_tensor(targets['keypoints'][np.newaxis], dtype=torch.float32)
        }
        # print(targets['labels'].shape, '\n', targets['boxes'].shape, '\n',targets['keypoints'].shape)

        return image, targets

# trainset = KeypointDataset('./train/images/', './train_data.json')
# trainset[0]
# import sys
# sys.exit()


transforms = A.Compose([
    ToTensorV2()],
    # bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
    # keypoint_params=A.KeypointParams(format='xy')
)

def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


trainset = KeypointDataset('./train/images/', './train_data.json', transforms)
train_loader = DataLoader(trainset, batch_size=12, shuffle=True, num_workers=6, collate_fn=collate_fn)

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

def train(device='cuda:0'):
    model = get_model()
    scaler = GradScaler()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with autocast():
                losses = model(images, targets)
                loss = sum(loss for loss in losses.values())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i+1) % 10 == 0:
                print(f'| epoch: {epoch} | loss: {loss.item():.4f}', end=' | ')
                for k, v in losses.items():
                    print(f'{k[5:]}: {v.item():.4f}', end=' | ')
                print()

        now = datetime.now()
        date = now.strftime('%d_%H_%M')
        print(epoch)
        torch.save(model.state_dict(), f'./keypoint-rcnn-{date}-{epoch}.pth')
        print(f'model{epoch} save!!!')

if __name__ == "__main__":
    train()