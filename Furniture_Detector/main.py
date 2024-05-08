import torch
import torchvision.transforms as transforms
from FurnitureDetector import get_furniture_detector
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
from Labels import labels
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Labels import labels
from Metadata import get_metadata
import os
import shutil

coco_labels = labels

good = []
bad = []

model = get_furniture_detector()
model.load_state_dict(torch.load("./FurnitureDetector_weights.pth"))
model.eval()

data = get_metadata("../dataset/metadata_good")
transform = transforms.Compose([transforms.ToTensor()])

for d in data:
    images = data[d].photos
    image_tensors = []
    image_pils = []
    for image in images:
        url = image.image
        path = os.path.join("../dataset/data", url)
        image = Image.open(path).convert("RGB")
        image_pils.append(image)
        image_tensors.append(transform(image))
    
    with torch.no_grad():
        predictions = model(image_tensors)

    print(d)
    for i, prediction in enumerate(predictions):
        draw = ImageDraw.Draw(image_pils[i])
        for box, label_id, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            label = coco_labels[int(label_id)]
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
            draw.text((box[0], box[1]), f"{label} ({score:.2f})", fill="red")
        image_pils[i].show()

    status = input("good or bad? [g,b]")

    if status == "g":
        good.append(d)
    elif status == "b":
        bad.append(d)
    else:
        print("bad input")

print(good)
print(bad)

# Array of local file paths
file_paths = [os.path.join("../dataset/metadata", g) for g in good]  # Add your file paths here

# Directory where you want to move the files
destination_directory = "../dataset/metadata_good"  # Replace with the desired destination directory

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Move each file to the destination directory
for file_path in file_paths:
    # Extract the filename from the file path
    file_name = os.path.basename(file_path)
    
    # Construct the destination path by joining the destination directory with the filename
    destination_path = os.path.join(destination_directory, file_name)
    
    # Move the file to the destination directory
    shutil.copy(file_path, destination_path)