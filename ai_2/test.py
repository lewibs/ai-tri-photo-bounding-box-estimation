import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
from FurnitureDetector import get_furniture_detector
from Anotations import load_anotations
import os

labels = [
    '__background__', 'object'
]

data_root = "../dataset/data_train"
basename = "0a8e9670-b1df-4a1e-9642-fa27556749f5.jpg"

# Step 1: Load pre-trained Faster R-CNN model
model = get_furniture_detector()

anotations = load_anotations("../dataset/anotations.csv")

anotation = anotations[basename]
print(anotation)

# Step 2: Load the image
image = Image.open(os.path.join(data_root, basename)).convert("RGB")

# # Step 3: Preprocess the image
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)

# # Step 4: Perform inference
# with torch.no_grad():
#     predictions = model([image_tensor])

# # Step 5: Display the image with bounding boxes and labels
draw = ImageDraw.Draw(image)
for anotation in anotation: 
    draw.rectangle([(anotation.box[0], anotation.box[1]), (anotation.box[2], anotation.box[3])], outline="red")
    draw.text((anotation.box[0], anotation.box[1]), f"{anotation.label})", fill="red")

image.show()