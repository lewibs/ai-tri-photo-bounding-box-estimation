import torch
from torchvision import tv_tensors
import os
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from Anotations import load_anotations
from Labels import label_to_index

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, anotations, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root))))
        self.anotations = load_anotations(anotations)

        bad_index = []
        bad_keys = []

        for index, key in enumerate(self.imgs):
            if key not in self.anotations:
                bad_index.append(index)
                bad_keys.append(key)

        self.imgs = [img for i, img in enumerate(self.imgs) if i not in bad_index]


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = read_image(img_path)

        num_objs = 1 #TODO this may change?

        anotation = self.anotations[self.imgs[idx]][0]

        image_id = idx
        boxes = anotation.box
        labels = anotation.label

        labels = [label_to_index(label) for label in labels]

        box = boxes
        height, width = img.shape[1:]
        box = torch.tensor(boxes) / 100
        box[3] *= width
        box[1] *= width
        box[0] *= height
        box[2] *= height

        area = width * height

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(box, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = image_id
        target["area"] = torch.tensor([area])
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)