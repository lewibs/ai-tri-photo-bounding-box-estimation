import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.io import read_image

def get_model(train=False):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 2  # 1 class (object) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if train == False:
        model.load_state_dict(torch.load("FurnitureDetector_weights.pth"))
        model.eval()

    return model

def get_image(image_path):
    # image = Image.open(image_path)
    image = read_image(image_path)
    image_tensor = get_transform(train=False)(image)
    image_tensor = image_tensor[:3, ...]
    return image, [image_tensor,]

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)