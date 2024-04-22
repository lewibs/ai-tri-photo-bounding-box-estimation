import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time

# Define your dataset paths and other parameters
train_data_dir = 'path/to/train/dataset'
val_data_dir = 'path/to/validation/dataset'
num_classes = 2  # Number of classes in your dataset (including background)
batch_size = 4
lr = 0.001
num_epochs = 10

# Define data transformations
train_transform = transforms.Compose([
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

# Load COCO train and validation datasets
train_dataset = CocoDetection(root=train_data_dir, annFile=train_annotation_file, transform=train_transform)
val_dataset = CocoDetection(root=val_data_dir, annFile=val_annotation_file, transform=val_transform)

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one that has the correct number of classes
num_classes = 2  # 1 (object) + 1 (background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define loss function and optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
# or use Adam optimizer
# optimizer = optim.Adam(params, lr=lr)

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for images, targets in train_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    # Print training statistics
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}, "
          f"Time: {time.time()-start_time}s")
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            val_loss_dict = model(images, targets)
            val_losses = sum(val_loss for val_loss in val_loss_dict.values())
    
    print(f"Validation Loss: {val_losses/len(val_dataloader)}")

# Save the trained model
torch.save(model.state_dict(), 'fine_tuned_faster_rcnn.pth')
