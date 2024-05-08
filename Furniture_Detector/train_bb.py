from Metadata import Dataset
from BBEstimator import BBEstimator
from Transformer import get_transform
import os
from load_image_from_file import load_image_from_file
import torch
from utils import collate_fn
import torch.utils.data as data
import utils
import torch.nn.functional as F

def main():
    NUM_EPOCHS = 4
    BATCH_SIZE = 2
    NUM_EPOCHS = 4
    NUM_WORKERS = 4
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP = 3
    GAMMA = 0.1
    CHECKPOINT = 'BBEstimator_weights.pth'

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #dont set train to true since all the images need to be their normal values to allow for
    #box detection
    # Define the length of training and validation sets
    dataset = Dataset("../dataset/metadata", "../dataset/data", get_transform(train=False))

    total_length = len(dataset)
    train_length = int(0.8 * total_length)  # 80% for training
    val_length = total_length - train_length  # Remaining for validation

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = data.random_split(dataset, [train_length, val_length])

    # Create data loaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = BBEstimator()
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

    for epoch in range(NUM_EPOCHS):
        print("Start epoch")
        #TRAIN
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=10)
        #UPDATE LEARNING RATE
        lr_scheduler.step()
        #EVALUATE
        test_loss = evaluate(model, val_loader, device=DEVICE)
        #SAVE STATUS
        torch.save(model.state_dict(), CHECKPOINT)
        print("Finished epoch")

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
        model.train()

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        for batch_idx, batch_data in enumerate(data_loader):
            fov, near, far, aspect, photos, positions, rotations, target = batch_data
            for i in range(len(fov)):
                prediction = model(fov[i], near[i], far[i], aspect[i], photos[i], positions[i], rotations[i])

                if prediction is None:
                    continue

                t = torch.tensor(target[i])
                loss = F.smooth_l1_loss(prediction, t)
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                print(f"iteration: {batch_idx}. Loss: {loss.item()}")
            

def evaluate(model, data_loader_test, device):
    print("penis")

if __name__ == "__main__":
    main()