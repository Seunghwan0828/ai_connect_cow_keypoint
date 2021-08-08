import json
import sys
from typing import Tuple, List, Sequence, Callable, Dict
import cv2
import numpy as np
import matplotlib.pyplot as plt



def draw_keypoints(
    image,
    keypoints,
    edges,
    keypoint_names, 
    boxes,
    dpi
) -> None:
    """
    Args:
        image (ndarray): [H, W, C]
        keypoints (ndarray): [N, 3]
        edges (List(Tuple(int, int))): 
    """

    with open('train_data.json', 'r') as anno_file:
        annt = json.load(anno_file)

    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(24)}

    if boxes is not None:
        x1, y1, x2, y2 = annt['annotation'][0]['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    for i, keypoint in enumerate(keypoints):
        cv2.circle(
            image, 
            tuple(keypoint), 
            3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image, 
                f'{i}: {keypoint_names[i]}', 
                tuple(keypoint), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image, 
                tuple(keypoints[edge[0]-1]), 
                tuple(keypoints[edge[1]-1]),
                colors.get(edge[0]), 3, lineType=cv2.LINE_AA)

    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image)
    ax.axis('off')
    plt.show()
    fig.savefig('example.png')


with open('train_data.json', 'r') as anno_file:
    annt = json.load(anno_file)

keypoints = annt['annotation'][0]['keypoints']
# print(keypoints)
keypoints = np.array(keypoints).reshape(17, -1)
keypoints = np.delete(keypoints, 2, axis=1)
# print(keypoints)
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

image = cv2.imread('./train/images/livestock_cow_keypoints_000001.jpg', cv2.COLOR_BGR2RGB)
draw_keypoints(image, keypoints, edges, keypoint_names, boxes=None, dpi=None)
