import torch
from Transformer import get_transform
import math
from FurnitureDetector import get_furniture_detector
import math
from engine import train_one_epoch, evaluate
from utils import collate_fn
from torchvision import transforms
from roboflow import Roboflow
from CustomCocoDataset import CustomCocoDataset

if __name__ == "__main__":
    BATCH_SIZE = 2
    NUM_EPOCHS = 4
    NUM_WORKERS = 4
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP = 3
    GAMMA = 0.1
    CHECKPOINT = 'FurnitureDetector_weights.pth'

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CLASSES = [
        '__background__', 'object'
    ]

    roboflow = Roboflow(api_key='pg3aiPpEDu1jgJpFSiGR')
    name = "furniture-objects"
    number = 1
    full_name = name + "-" + str(number)

    project = roboflow.workspace("lewibser-ssm5b").project(name)
    version = project.version(number)
    dataset = version.download("coco")

    dataset = CustomCocoDataset(root="./furniture-objects-1/train", annFile="./furniture-objects-1/train/_annotations.coco.json", transform=get_transform(train=True))
    dataset_test = CustomCocoDataset(root="./furniture-objects-1/test", annFile="./furniture-objects-1/test/_annotations.coco.json", transform=get_transform(train=False))

    # dataset = Dataset(
    #     root="../dataset/data_train",
    #     anotations="../dataset/anotations.csv",
    #     transforms=get_transform(train=True),
    # )

    # dataset_test = Dataset(
    #     root="../dataset/data_train",
    #     anotations="../dataset/anotations.csv",
    #     transforms=get_transform(train=False),
    # )

    # dataset = PennFudanDataset_LabelStudio(
    #     root="../dataset/PennFudanPed/PNGImages",
    #     anotations="../dataset/PennFudanPed/anotations.csv",
    #     transforms=get_transform(train=True),
    # )

    # dataset_test = PennFudanDataset_LabelStudio(
    #     root="../dataset/PennFudanPed/PNGImages",
    #     anotations="../dataset/PennFudanPed/anotations.csv",
    #     transforms=get_transform(train=False),
    # )

    # dataset = PennFudanDataset('../dataset/PennFudanPed', get_transform(train=True))
    # dataset_test = PennFudanDataset('../dataset/PennFudanPed', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = get_furniture_detector()
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP,
        gamma=GAMMA
    )

    # test and visualization here
    # images, targets = next(iter(data_loader))
    # images = list(image for image in images)
    # targets = [{k:v for k, v in t.items()} for t in targets]
    # print(images, targets)
    # output = model(images, targets)
    # print(output)

    for epoch in range(NUM_EPOCHS):
        print("Start epoch")
        #TRAIN
        train_loss = train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=10)
        #UPDATE LEARNING RATE
        lr_scheduler.step()
        #EVALUATE
        test_loss = evaluate(model, data_loader_test, device=DEVICE)
        #SAVE STATUS
        torch.save(model.state_dict(), CHECKPOINT)
        print("Finished epoch")