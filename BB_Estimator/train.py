import Metadata
import os

root = "../dataset/metadata"
data = Metadata.get_metadata(root)[0]

for photo in data["photos"]:
    print(Metadata.get_image(os.path.join(root, photo["image"])))