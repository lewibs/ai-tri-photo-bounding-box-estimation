import os
import json
from collections import namedtuple
from torchvision.io import read_image

Metadata = namedtuple("Metadata", ["target", "camera", "photos"])
Target = namedtuple("Target", ["box", "format"])
Vector = namedtuple("Vector", ["vector", "format"])
Camera = namedtuple("Camera", ["fov", "aspect", "near", "far"])
Image = namedtuple("Image", ["position", "rotation", "image"])

def get_metadata(root):    
    files = os.listdir(root)

    output = {}

    for file in files:
        with open(os.path.join(root, file), "r") as json_file:
            data = json.load(json_file)

        target = Target([
            data["bounding_box"]["min"]["x"],
            data["bounding_box"]["min"]["y"],
            data["bounding_box"]["min"]["z"],
            data["bounding_box"]["max"]["x"],
            data["bounding_box"]["max"]["y"],
            data["bounding_box"]["max"]["z"],
        ], "XYZXYZ")

        camera = Camera(data["fov"], data["aspect"], data["near"], data["far"])

        photos = []
        for photo in data["photos"]:
            photos.append(Image(Vector(photo["position"], "XYZ"), Vector(photo["rotation"][:-1], "XYZ"), photo["image"]))

        output[file] = Metadata(target, camera, photos)
        
    return output

def get_image(file):
    return read_image(file)