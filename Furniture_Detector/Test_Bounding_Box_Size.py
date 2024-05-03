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

# Step 1: Load pre-trained Faster R-CNN model
model = get_furniture_detector()

anotations = load_anotations("../dataset/anotations.csv")

i = 0

while True:
    # Get the index from the user
    index = input("Enter to see the next image: ")
    
    # Check if the user pressed Enter to exit
    if index != "":
        break
    
    try:
        basename = list(anotations.keys())[i]
        annotation = anotations[basename]

        # Load the image
        image = Image.open(os.path.join(data_root, basename)).convert("RGB")

        # Preprocess the image
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)

        # Get the width and height of the image
        width, height = image.size

        # Perform inference (disabled for now)
        # with torch.no_grad():
        #     predictions = model([image_tensor])
        #     print(predictions)

        # Display the image with bounding boxes and labels
        draw = ImageDraw.Draw(image)
        for annotation in annotation:
            x_min, y_min, x_max, y_max = annotation.box
            dims = [(width * (x_min / 100), height * (y_min / 100)), (width * (x_max / 100), height * (y_max / 100))]
            draw.rectangle(dims, outline="red")
            draw.text((width * (x_min / 100), height * (y_min / 100)), f"{annotation.label}", fill="red")

        image.show()
        i += 1
    except (ValueError, IndexError):
        print("Invalid index. Please enter a valid index within the range.")