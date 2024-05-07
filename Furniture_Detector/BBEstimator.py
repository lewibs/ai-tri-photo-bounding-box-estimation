import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Furniture_Detector.FurnitureDetector import get_furniture_detector

class BBEstimator(nn.Module):
    def __init__(self):
        super(BBEstimator, self).__init__()

        self.image_model = get_furniture_detector()
        

        # self.layers = nn.Sequential(

        # )

        # self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer from input to hidden
        # self.fc2 = nn.Linear(hidden_size, output_size) # Fully connected layer from hidden to output

    def forward(self, fov, near, far, aspect, photos, positions, rotations):
        photo_predictions = self.image_model(photos)
        
        box_predictions = []
        for d in photo_predictions:
            max_score_idx = d['scores'].argmax().item()  # Find index of maximum score
            max_score_box = d['boxes'][max_score_idx]
            box_predictions.append(max_score_box)

        print(box_predictions)

        return None

# Example usage:
input_size = 10
hidden_size = 20
output_size = 1  # Assuming it's a regression problem with one output value

model = BBEstimator(input_size, hidden_size, output_size)
print(model)
