import json
import argparse
from tqdm import tqdm

categories_seen = [
    {'id': 1, 'name': 'person'},
    {'id': 2, 'name': 'bicycle'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'motorcycle'},
    {'id': 7, 'name': 'train'},
    {'id': 8, 'name': 'truck'},
    {'id': 9, 'name': 'boat'},
    {'id': 15, 'name': 'bench'},
    {'id': 16, 'name': 'bird'},
    {'id': 19, 'name': 'horse'},
    {'id': 20, 'name': 'sheep'},
    {'id': 23, 'name': 'bear'},
    {'id': 24, 'name': 'zebra'},
    {'id': 25, 'name': 'giraffe'},
    {'id': 27, 'name': 'backpack'},
    {'id': 31, 'name': 'handbag'},
    {'id': 33, 'name': 'suitcase'},
    {'id': 34, 'name': 'frisbee'},
    {'id': 35, 'name': 'skis'},
    {'id': 38, 'name': 'kite'},
    {'id': 42, 'name': 'surfboard'},
    {'id': 44, 'name': 'bottle'},
    {'id': 48, 'name': 'fork'},
    {'id': 50, 'name': 'spoon'},
    {'id': 51, 'name': 'bowl'},
    {'id': 52, 'name': 'banana'},
    {'id': 53, 'name': 'apple'},
    {'id': 54, 'name': 'sandwich'},
    {'id': 55, 'name': 'orange'},
    {'id': 56, 'name': 'broccoli'},
    {'id': 57, 'name': 'carrot'},
    {'id': 59, 'name': 'pizza'},
    {'id': 60, 'name': 'donut'},
    {'id': 62, 'name': 'chair'},
    {'id': 65, 'name': 'bed'},
    {'id': 70, 'name': 'toilet'},
    {'id': 72, 'name': 'tv'},
    {'id': 73, 'name': 'laptop'},
    {'id': 74, 'name': 'mouse'},
    {'id': 75, 'name': 'remote'},
    {'id': 78, 'name': 'microwave'},
    {'id': 79, 'name': 'oven'},
    {'id': 80, 'name': 'toaster'},
    {'id': 82, 'name': 'refrigerator'},
    {'id': 84, 'name': 'book'},
    {'id': 85, 'name': 'clock'},
    {'id': 86, 'name': 'vase'},
    {'id': 90, 'name': 'toothbrush'},
]

base_cat_ids = [cat['id'] for cat in categories_seen]

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="data/coco/annotations/instances_train2017.json", type=str)
parser.add_argument("--out_path", default="data/coco/wusize/instances_train2017_base.json")
args = parser.parse_args()

with open(args.json_path, 'r') as f:
    json_coco = json.load(f)

annotations = []

for ann in tqdm(json_coco['annotations']):
    if ann['category_id'] in base_cat_ids:
        annotations.append(ann)

json_coco['annotations'] = annotations

with open(args.out_path, 'w') as f:
    json.dump(json_coco, f)
