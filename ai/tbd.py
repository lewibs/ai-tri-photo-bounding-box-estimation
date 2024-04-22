import os
import json
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

directory = "../dataset/metadata"
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

good = []
bad = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):  # Check if the current item is a file
        # Open and process the file
        with open(filepath, 'r') as file:
            # Perform operations on the file
            # For example, read the content of the file
            content = json.load(file)
            images = []

            for photo in content["photos"]:
                url = photo["image"]
                image = Image.open(f"../dataset/data/{url}")
                image_tensor = F.to_tensor(image).unsqueeze(0)
                
                with torch.no_grad():
                    predictions = model(image_tensor)

                # Display the results
                draw = ImageDraw.Draw(image)
                for box, label, score in zip(predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["scores"]):
                    if score > 0.5:  # Filter out low-confidence detections
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
                        draw.text((box[0], box[1]), f"{label.item()}", fill="red")

                images.append(image)
            
            # show the images here all at once
            num_images = len(images)
            fig, axes = plt.subplots(1, num_images, figsize=(10*num_images, 10))
            for i, img in enumerate(images):
                axes[i].imshow(img)
                axes[i].axis('off')
            plt.show()

    flag = input("Good or Bad: (g,b): ")
    if flag == "g":
        good.append(content["id"])
    else:
        bad.append(content["id"])


# Save the "good" array into a JSON file
with open("good.json", "w") as f:
    json.dump(good, f)

# Save the "bad" array into a JSON file
with open("bad.json", "w") as f:
    json.dump(bad, f)


                


