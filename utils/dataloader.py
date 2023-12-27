import mmcv
import numpy as np
import os
import os.path as osp
import json
import imagesize
import pickle

from .builder import DATASETS
from .custom import CustomDataset
### This is for how to transfer current format into COCOdataset format
LABEL_MAPPING_VER_1_16 = {
    'car': 0,
    'policecar': 0,
    'bus': 1,
    'fireengine': 2,
    'truck': 2,
    'heavy_equipment': 2,
    'heavy-equipment': 2,
    'heavyequipment': 2,
    'flatbedtrailertruck': 2,
    'trailer': 2,
    'moto': 3,
    "motorcycle": 3,
    'pedestrian': 4,
    'ped': 4,
    'driver': 4,
    'rider': 4,
    'bike': 5,
    'cyclist': 5,
    "bicycle": 5,
    'traffic_cone': 6,
    'cone': 6,
    'barricade': 7,
    'movable_traffic_sign': 7,
    'movabletrafficsign': 7,
    'suv': 8,    # 'suv + van'
    'van': 8,
    'ambulance': 8,
    'lighttruck': 9,   # 'lighttruck + pickup'
    'pickup': 9,
}

INVERSE_LABEL_MAPPING_VER_1_16 = [
    'car',
    'bus',
    'truck',
    'moto',
    'pedestrian',
    'bike',
    'traffic_cone',
    'barricade',
    'suv',
    'lighttruck',
]


@DATASETS.register_module()
class PlusDataset(CustomDataset):
    LABEL_MAPPING = LABEL_MAPPING_VER_1_16
    CLASSES = INVERSE_LABEL_MAPPING_VER_1_16
    
    def load_annotations(self, ann_file):
        train_presave_path = 'data/plus/plus_det_train.pkl'
        eval_presave_path = 'data/plus/plus_det_eval.pkl'
        if self.test_mode:       
            if os.path.exists(eval_presave_path):
                with open(eval_presave_path, 'rb') as f:
                    data_infos = pickle.load(f)
                return data_infos
        else:
            if os.path.exists(train_presave_path):
                with open(train_presave_path, 'rb') as f:
                    data_infos = pickle.load(f)
                return data_infos            
            
        
        data_infos = []
        print('mode: ', self.test_mode)
        print('ann_file: ', ann_file)
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        for img in data:
            boxes = []
            labels = []
            img_path = osp.join(self.img_prefix, img['filename'])
            if not osp.exists(img_path):
                continue
            if 'image_width' in img and 'image_height' in img \
                    and int(img['image_width']) > 0 and int(img['image_height']) > 0:
                width = int(img['image_width'])
                height = int(img['image_height'])
            else:
                width, height = imagesize.get(img_path)
            
            for anno_idx, anno in enumerate(img['annotations']):
                if anno['class'] not in self.LABEL_MAPPING:
                    continue
                x = max(0, anno['x'])
                y = max(0, anno['y'])
                if ('width' not in anno) or ('height' not in anno):
                    print("Labeling error! Width or height unavailable.")
                    print("json file=", ann_file)
                    continue
                w = anno['width']
                h = anno['height']

                box = [x, y, min(x + w, width - 1), min(y + h, height - 1)]
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                if box_area <= 1:
                    continue
                
                boxes.append(np.array(box))
                labels.append(self.LABEL_MAPPING[anno['class']])   
            
            data_infos.append(
                dict(
                    filename=img['filename'],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(boxes).astype(np.float32).reshape(-1, 4),
                        labels=np.array(labels).astype(np.int64)
                    )
                )
            )
        
        if self.test_mode:
            if not os.path.exists(eval_presave_path):
                with open(eval_presave_path, 'wb') as f:
                    pickle.dump(data_infos, f)
        else:
            if not os.path.exists(train_presave_path):
                with open(train_presave_path, 'wb') as f:
                    pickle.dump(data_infos, f)            
        print(len(data_infos))
        return data_infos                   
                
            
                     