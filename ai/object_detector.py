import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
import pandas as pd
import json
import utils
from engine import train_one_epoch, evaluate

# Made with adaptions from this toutorial:
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, boxes, labels, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root))))
        self.boxes = boxes
        self.lables = labels

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
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def get_model():
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 2  # 1 class (object) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def downloadStuff():
    # URLs of the files to download
    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
    ]

    # Destination directory where the files will be saved
    destination_dir = "../ai"

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Download each file
    for url in urls:
        filename = os.path.join(destination_dir, os.path.basename(url))
        os.system(f"powershell Invoke-WebRequest -Uri {url} -OutFile {filename}")

    # Verify if files are downloaded successfully
    for filename in os.listdir(destination_dir):
        print(filename)

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


if __name__ == "__main__":
    downloadStuff()
    csv_file = '../dataset/anotations.csv'

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    ids = df.values[:, 4]
    metadata = df.values[:, 5]
    boxes = {}
    labels = {}

    for i, data in enumerate(metadata):
        data = json.loads(data)
        basename = os.path.basename(ids[i])
        # Split the string using the '-' delimiter
        basename = basename.split('-')

        # Remove the first element from the resulting list
        basename = basename[1:]

        # Join the remaining parts back together using the '-' delimiter
        basename = '-'.join(basename)

        for data in data:
            #TODO this can cause overwrite issues but because there is only one box per image im not going to deal with it
            boxes[basename] = [data["x"], data["y"], data["x"] + data["width"], data["y"] + data["height"]]
            labels[basename] = data["rectanglelabels"]

    # def __init__(self, root, boxes, labels, transforms):
    dataset = Dataset(
        root="../dataset/data_train",
        boxes=boxes,
        labels=labels,
        transforms=get_transform(train=True)    
    )
    dataset_test = Dataset(
        root="../dataset/data_train",
        boxes=boxes,
        labels=labels,
        transforms=get_transform(train=False)    
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    model = get_model()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 2

    # Define paths for saving the model checkpoints
    checkpoint_path = 'model_checkpoint.pth'

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        test_loss = evaluate(model, data_loader_test, device=device)

        # Save model checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }, checkpoint_path)