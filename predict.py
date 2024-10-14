import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def load_attr_names(attr_cloth_file):
    with open(attr_cloth_file, 'r') as f:
        lines = f.readlines()[2:]  # Skip first two lines
        attr_names = [line.strip().split()[0] for line in lines]
    return attr_names

def predict_attributes(model_path, img_path, attr_cloth_file):
    # Load the attribute names
    attr_names = load_attr_names(attr_cloth_file)

    img_size=224
    # Load the trained model
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1000)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    # Prepare the image
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    img = Image.open(img_path).convert('RGB')
    #img_tensor = transform(img).unsqueeze(0)
    img.thumbnail((img_size, img_size), Image.LANCZOS)
    img = transform(img).unsqueeze(0)
    # Make prediction
    with torch.no_grad():
        output = model(img)
        logits = output.squeeze().numpy()

    # Convert predictions to binary (0 or 1)
    binary_predictions = (logits > -2).astype(int)
    #print(binary_predictions)
    # Create the binary string
    binary_string = ''.join(map(str, binary_predictions))

    # Get the positive predictions and their corresponding attribute names
    positive_attrs = [(attr_names[i], i) for i, pred in enumerate(binary_string) if int(pred) == 1]
    # for i, pred in enumerate(binary_string):
    #     if int(pred)==1:
    #         print("hit")

    return binary_string, positive_attrs

# Usage
model_path = 'DFResNet_3UF_300kIMG.pth'
img_path = 'myTest/750352725.jpg'
attr_cloth_file = 'Anno/list_attr_cloth.txt'

binary_string, positive_attrs = predict_attributes(model_path, img_path, attr_cloth_file)
#
#print("Binary prediction string:")
#print(binary_string)
print("\nPositive predictions:")
#for attr, index in positive_attrs:
#    print(f"{attr} (index: {index})")
print(positive_attrs)