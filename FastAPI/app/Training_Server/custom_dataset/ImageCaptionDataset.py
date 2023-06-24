import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import os
import math
import json
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

class ImageCaptionDataset(Dataset):
    def __init__(self, annotations_path, processor):
        # annotations = dict_keys(['info', 'images', 'licenses', 'annotations'])
        self.mode = os.path.splitext(os.path.basename(annotations_path))[0].split('_')[-1]
        self.json_array = json.load(open(annotations_path))
        self.dataset = {
            'image_paths' : [],
            'captions' : []
        }
        # sort forr indexing
        for image_json, annotation_json in zip(sorted(self.json_array['images'], key=lambda x : x['id']), sorted(self.json_array['annotations'], key=lambda x : x['image_id'])):
            # image_paths = Path(f'{self.mode}/{self.mode}' + image_json['file_name'])
            # captions = annotation_json['captions']
            
            # add image_paths, caption for indexing
            self.dataset['image_paths'].append(Path(f'cocoDataset/{self.mode}/{self.mode}/' + image_json['file_name']))
            self.dataset['captions'].append(annotation_json['caption']) 
        
        # prepare processor for Natural Language Preprocessing
        self.processor = processor

        
    def __len__(self):
        return len(self.dataset['image_paths'])
    
    def __getitem__(self, idx):
        image = Image.open(self.dataset['image_paths'][idx])
        caption = self.dataset['captions'][idx]
        # 사용할 processor가 batch 단위를 생각하고 encoding을 하기 때문에 squeeze()를 해야함.
        encoding = self.processor(images=image, text=caption, padding="max_length", return_tensors="pt")

        encoding = {k : v.squeeze() for k, v in encoding.items()}
        return encoding