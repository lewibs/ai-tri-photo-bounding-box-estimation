import torch
import torch.nn as nn
import torch.nn.functional as F
from FurnitureDetector import get_furniture_detector

class BBEstimator(nn.Module):
    def __init__(self):
        super(BBEstimator, self).__init__()

        self.image_model = get_furniture_detector()
        self.image_model.load_state_dict(torch.load("./FurnitureDetector_weights.pth"))
        self.image_model.eval()

        self.layers = nn.Sequential(
            nn.Linear(34, 64),  # Input size: 34, Output size: 64
            nn.ReLU(),           # Apply ReLU activation function
            nn.Linear(64, 32),   # Input size: 64, Output size: 32
            nn.ReLU(),           # Apply ReLU activation function
            nn.Linear(32, 6)     # Input size: 32, Output size: 6 (XYZXYZ)
        )

    def forward(self, fov, near, far, aspect, photos, positions, rotations):
        self.image_model.eval()
        photo_predictions = self.image_model(photos)

        if len(photo_predictions) == 0:
            return None

        
        box_predictions = []
        for d in photo_predictions:
            max_score_idx = d['scores'].argmax().item()  # Find index of maximum score
            max_score_box = d['boxes'][max_score_idx]
            box_predictions.append(max_score_box)

        box_predictions = torch.stack(box_predictions)
        positions = torch.tensor(positions)
        rotations = torch.tensor(rotations)

        box_predictions = box_predictions.flatten() #12
        positions = positions.flatten() #9
        rotations = positions.flatten() #9

        input = torch.cat((torch.tensor([fov]),torch.tensor([near]),torch.tensor([far]), torch.tensor([aspect]), positions, rotations, box_predictions))

        x = self.layers(input)

        return x
