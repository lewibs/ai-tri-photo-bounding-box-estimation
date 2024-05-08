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

coco_labels = labels


# Step 1: Load pre-trained Faster R-CNN model

model = get_furniture_detector()
model.load_state_dict(torch.load("./FurnitureDetector_weights.pth"))
model.eval()

# Step 2: Load the image
image_path = "../dataset/data_train/ffafd196-2f0e-430b-92b1-5902466c5f34.jpg"
image = Image.open(image_path).convert("RGB")
image_path2 = "../dataset/data_train/ffafd196-2f0e-430b-92b1-5902466c5f34.jpg"
image2 = Image.open(image_path2).convert("RGB")


# Step 3: Preprocess the image
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)
image_tensor2 = transform(image2)

# Step 4: Perform inference
with torch.no_grad():
    predictions = model([image_tensor, image_tensor2])
    print(predictions)

# Step 5: Display the image with bounding boxes and labels
draw = ImageDraw.Draw(image)
for box, label_id, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    label = coco_labels[int(label_id)]
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
    draw.text((box[0], box[1]), f"{label} ({score:.2f})", fill="red")

image.show()