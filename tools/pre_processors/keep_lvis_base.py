import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="data/lvis_v1/annotations/lvis_v1_train.json")
parser.add_argument("--out_path", default="data/lvis_v1/wusize/lvis_v1_train_base.json")
args = parser.parse_args()

with open(args.json_path, 'r') as f:
    json_coco = json.load(f)

annotations = []


cat_id2cat_info = {cat_info['id']: cat_info for cat_info in json_coco['categories']}
for ann in tqdm(json_coco['annotations']):
    cat_id = ann['category_id']
    cat_info = cat_id2cat_info[cat_id]
    frequency = cat_info['frequency']
    if frequency in ['f', 'c']:
        annotations.append(ann)

json_coco['annotations'] = annotations

with open(args.out_path, 'w') as f:
    json.dump(json_coco, f)
