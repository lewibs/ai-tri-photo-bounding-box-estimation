import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from Anotations import load_anotations


class PennFudanDataset_LabelStudio(torch.utils.data.Dataset):
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

        anotations = self.anotations[self.imgs[idx]]

        image_id = idx
        boxes = [anotation.box for anotation in anotations]

        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        image_id = idx

        area = torch.tensor([(box[3] - box[1]) * (box[2] - box[0]) for box in boxes])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        # target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)