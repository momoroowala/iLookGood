import numpy as np
from torch.utils.data import DataLoader
from Attr_Pred import AttrDataset
import torch

def parse_partition_file(partition_file):
    with open(partition_file, 'r') as f:
        partitions = f.readlines()[2:]  # Skip first two lines
    train_list, val_list, test_list = [], [], []
    for line in partitions:
        img_name, partition = line.strip().split()
        if partition == 'train':
            train_list.append(img_name)
        elif partition == 'val':
            val_list.append(img_name)
        elif partition == 'test':
            test_list.append(img_name)
    return train_list, val_list, test_list

def create_img_file(img_list, output_file):
    with open(output_file, 'w') as f:
        for img_name in img_list:
            f.write(f"{img_name}\n")

# Parse partition file
train_list, val_list, test_list = parse_partition_file('Eval/list_eval_partition.txt')

# Create temporary img files for each partition
create_img_file(train_list, 'train_img_file.txt')
create_img_file(val_list, 'val_img_file.txt')
create_img_file(test_list, 'test_img_file.txt')
# Create datasets
train_dataset = AttrDataset(
    img_path='img/',
    img_file='train_img_file.txt',
    attr_cloth_file='Anno/list_attr_cloth.txt',
    attr_img_file='Anno/list_attr_img.txt',
    category_cloth_file='Anno/list_category_cloth.txt',
    category_img_file='Anno/list_category_img.txt',
    bbox_file='Anno/list_bbox.txt',
    #landmark_file='Anno/list_landmarks.txt',
    img_size=(224, 224)
    #subset_size=50000  # Adjust as needed
)
print("Train Data Organized")
val_dataset = AttrDataset(
    img_path='img/',
    img_file='val_img_file.txt',
    attr_cloth_file='Anno/list_attr_cloth.txt',
    attr_img_file='Anno/list_attr_img.txt',
    category_cloth_file='Anno/list_category_cloth.txt',
    category_img_file='Anno/list_category_img.txt',
    bbox_file='Anno/list_bbox.txt',
   # landmark_file='Anno/list_landmarks.txt',
    img_size=(224, 224)
    #subset_size=10000  # Adjust as needed
)
print("Val Data Organized")
test_dataset = AttrDataset(
    img_path='img/',
    img_file='test_img_file.txt',
    attr_cloth_file='Anno/list_attr_cloth.txt',
    attr_img_file='Anno/list_attr_img.txt',
    category_cloth_file='Anno/list_category_cloth.txt',
    category_img_file='Anno/list_category_img.txt',
    bbox_file='Anno/list_bbox.txt',
  #  landmark_file='Anno/list_landmarks.txt',
    img_size=(224, 224)
    #subset_size=10000  # Adjust as needed
)
print("Test Data Organized")
torch.save(train_dataset, 'train_dataset.pt')
print("Train Data Saved")
torch.save(val_dataset, 'val_dataset.pt')
print("Val Data Saved")
torch.save(test_dataset, 'test_dataset.pt')
print("Test Data Saved")
# Create dataloaders
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
#print("Train Data Loaded")
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
#print("Train Data Loaded")
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
#print("Train Data Loaded")