import torch
from Dataset import Dataset
from Transformer import get_transform
import math
from FurnitureDetector import get_furniture_detector
import math
from engine import train_one_epoch, evaluate
from utils import collate_fn

if __name__ == "__main__":
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    NUM_WORKERS = 4
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP = 3
    GAMMA = 0.1
    CHECKPOINT = 'FurnitureDetector_weights.pth'
    TRAINING_SPLIT = 0.80

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CLASSES = [
        '__background__', 'object'
    ]

    NUM_CLASSES = len(CLASSES)

    # location to save model and plots
    OUT_DIR = 'outputs'

    dataset = Dataset(
        root="../dataset/data_train",
        anotations="../dataset/anotations.csv",
        transforms=get_transform(train=True),
    )

    dataset_test = Dataset(
        root="../dataset/data_train",
        anotations="../dataset/anotations.csv",
        transforms=get_transform(train=False),
    )

    indices = torch.randperm(len(dataset)).tolist()
    images = len (indices)

    dataset_count = math.floor(images*TRAINING_SPLIT)
    dataset_test_count = math.floor(images*(1-TRAINING_SPLIT))

    dataset = torch.utils.data.Subset(dataset, indices[-dataset_count:])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:dataset_test_count])

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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = get_furniture_detector()

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