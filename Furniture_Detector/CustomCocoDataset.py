import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.coco = CocoDetection(root, annFile)
        self.transform = transform

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        image, coco_target = self.coco[idx]
        if self.transform:
            image = self.transform(image)

        boxes = []
        labels = []
        for target in coco_target:
            labels.append(target["category_id"])
            x = target["bbox"][0]
            y = target["bbox"][1]
            width = target["bbox"][2]
            height = target["bbox"][3]
            boxes.append([x,y,x+width,y+height])

        image = self.transform(tv_tensors.Image(image))
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        # tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = torch.tensor(labels)
        target["image_id"] = target["image_id"]
        target["area"] = torch.tensor(target["area"])
        target["iscrowd"] = torch.zeros((len(labels),))

        return image, target