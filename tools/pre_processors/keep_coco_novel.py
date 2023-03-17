import json
import argparse
from tqdm import tqdm

categories_unseen = [
    {'id': 5, 'name': 'airplane'},
    {'id': 6, 'name': 'bus'},
    {'id': 17, 'name': 'cat'},
    {'id': 18, 'name': 'dog'},
    {'id': 21, 'name': 'cow'},
    {'id': 22, 'name': 'elephant'},
    {'id': 28, 'name': 'umbrella'},
    {'id': 32, 'name': 'tie'},
    {'id': 36, 'name': 'snowboard'},
    {'id': 41, 'name': 'skateboard'},
    {'id': 47, 'name': 'cup'},
    {'id': 49, 'name': 'knife'},
    {'id': 61, 'name': 'cake'},
    {'id': 63, 'name': 'couch'},
    {'id': 76, 'name': 'keyboard'},
    {'id': 81, 'name': 'sink'},
    {'id': 87, 'name': 'scissors'},
]

novel_cat_ids = [cat['id'] for cat in categories_unseen]

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="data/coco/annotations/instances_val2017.json")
parser.add_argument("--out_path", default="data/coco/wusize/instances_val2017_novel.json")
args = parser.parse_args()

with open(args.json_path, 'r') as f:
    json_coco = json.load(f)

annotations = []

for ann in tqdm(json_coco['annotations']):
    if ann['category_id'] in novel_cat_ids:
        annotations.append(ann)

json_coco['annotations'] = annotations

with open(args.out_path, 'w') as f:
    json.dump(json_coco, f)
