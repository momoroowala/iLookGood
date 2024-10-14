from __future__ import division
import os

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

#from .registry import DATASETS


#@DATASETS.register_module
class AttrDataset(Dataset):
    CLASSES = None
    def __init__(self,
                 img_path,
                 img_file,
                 attr_cloth_file,
                 attr_img_file,
                 category_cloth_file,
                 category_img_file,
                 bbox_file,
                 #landmark_file,
                 img_size,
                 subset_size=None,
                 idx2id=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
        self.img_path = img_path
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # read img names
        with open(img_file, 'r') as fp:
            self.img_list = [x.strip() for x in fp.readlines()]

        # read attribute names
        with open(attr_cloth_file, 'r') as f:
            lines = f.readlines()[2:]  # Skip first two lines
            self.attr_names = [line.strip().split()[0] for line in lines]

        # read attribute labels
        with open(attr_img_file, 'r') as f:
            lines = f.readlines()[2:]  # Skip first two lines
            self.attr_labels = {}
            for line in lines:
                parts = line.strip().split()
                img_name = parts[0]
                attrs = [int(x) for x in parts[1:]]
                self.attr_labels[img_name] = attrs

        # read category names
        with open(category_cloth_file, 'r') as f:
            lines = f.readlines()[2:]  # Skip first two lines
            self.category_names = [line.strip().split()[0] for line in lines]

        # read category labels
        with open(category_img_file, 'r') as f:
            lines = f.readlines()[2:]  # Skip first two lines
            self.category_labels = {}
            for line in lines:
                parts = line.strip().split()
                img_name = parts[0]
                category = int(parts[1]) - 1  # Subtract 1 to make it 0-indexed
                self.category_labels[img_name] = category

        self.img_size = img_size
        # load bbox
        if bbox_file:
            self.with_bbox = True
            self.bboxes = np.loadtxt(bbox_file, skiprows=2, usecols=(1, 2, 3, 4))
        else:
            self.with_bbox = False
            self.bboxes = None
        # load landmarks
       # if landmark_file:
        #    self.landmarks = np.loadtxt(landmark_file, skiprows=2)[:, 2:]  # Skip image name and clothing type
        #else:
        #    self.landmarks = None
        if subset_size is not None and subset_size < len(self.img_list):
            ## HIGHLIGHT: Use stratified sampling to maintain label distribution
            self.img_list = self.stratified_sample(self.img_list, subset_size)

    def stratified_sample(self, img_list, subset_size):
        label_to_imgs = {}
        for img in img_list:
            label = tuple(self.attr_labels[img])
            if label not in label_to_imgs:
                label_to_imgs[label] = []
            label_to_imgs[label].append(img)
        
        sampled_imgs = []
        while len(sampled_imgs) < subset_size:
            for label in label_to_imgs:
                if label_to_imgs[label]:
                    sampled_imgs.append(label_to_imgs[label].pop())
                if len(sampled_imgs) == subset_size:
                    break
        
        return sampled_imgs

    def get_basic_item(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        width, height = img.size
        if self.with_bbox:
            bbox_cor = self.bboxes[idx]
            x1, y1, x2, y2 = bbox_cor
            x1, y1, x2, y2 = max(0, x1-10), max(0, y1-10), x2+10, y2+10
            bbox_w, bbox_h = x2-x1, y2-y1
            img = img.crop((x1, y1, x2, y2))
        else:
            bbox_w, bbox_h = self.img_size[0], self.img_size[1]
        img.thumbnail(self.img_size, Image.LANCZOS)
        img = self.transform(img)
        
        # Get attribute labels for this image
        attr_label = self.attr_labels.get(img_name, [0] * len(self.attr_names))
        label = torch.tensor(attr_label, dtype=torch.float32)
        
        # Get category for this image
        # category = self.category_labels.get(img_name, 0)  # Default to 0 if not found
        # cate = torch.tensor([category], dtype=torch.long)
        
       # landmark = []
        # compute the shiftness
       # if self.landmarks is not None:
       #     origin_landmark = self.landmarks[idx]
        #    for i, l in enumerate(origin_landmark):
        #        if i % 2 == 0:  # x
       #             l_x = max(0, l - x1)
        #            l_x = float(l_x) / bbox_w * self.img_size[0]
       #             landmark.append(l_x)
        #        else:  # y
        #            l_y = max(0, l - y1)
       #             l_y = float(l_y) / bbox_h * self.img_size[1]
       #             landmark.append(l_y)
       #     landmark = torch.tensor(landmark, dtype=torch.float32)
       # else:
       #     landmark = torch.zeros(8)
        
        # data = {'img': img, 'attr': label, 'cate': cate, 'landmark': landmark}
        # data = {'img': img, 'attr': label, 'cate': cate}
        data = {'img': img, 'attr': label}
        return data

    def __getitem__(self, idx):
        data = self.get_basic_item(idx)
        return data['img'], data['attr']

    def __len__(self):
        return len(self.img_list)