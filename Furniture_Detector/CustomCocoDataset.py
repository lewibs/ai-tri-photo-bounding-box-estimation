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
        image, coco_target = self.coco.__getitem__(idx)

        if len(coco_target) == 0:
            image, coco_target = self.coco.__getitem__(idx - 1) #TODO dont just cheat like this when the coco is null
            #TODO coco is null when there are no annotations in the image

        try:
            coco_target = coco_target[0] #TODO
        except:
            print(image, coco_target)

        boxes = []
        labels = []
        
        labels.append(coco_target["category_id"])
        x = coco_target["bbox"][0]
        y = coco_target["bbox"][1]
        width = coco_target["bbox"][2]
        height = coco_target["bbox"][3]
        boxes.append([x,y,x+width,y+height])
        
        if self.transform:
            image = self.transform(tv_tensors.Image(image))
        
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        # tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = torch.tensor(labels)
        target["image_id"] = coco_target["image_id"]
        target["area"] = torch.tensor([coco_target["area"]])
        target["iscrowd"] = torch.zeros((len(labels),))

        return image, target