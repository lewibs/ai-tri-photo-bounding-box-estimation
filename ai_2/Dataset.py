import torch
from torchvision import tv_tensors
import os
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from Anotations import load_anotations

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, anotations, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root))))
        self.anotation = load_anotations(anotations) 

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = read_image(img_path)

        num_objs = 1

        boxes = self.boxes[self.imgs[idx]]
        labels = self.lables[self.imgs[idx]]

        image_id = idx
        area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = torch.tensor([1], dtype=torch.int64)
        target["image_id"] = image_id
        target["area"] = torch.tensor([area])
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)