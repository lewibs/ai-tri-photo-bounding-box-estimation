import os
import json
from collections import namedtuple
from torchvision.io import read_image
import torch
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from load_image_from_file import load_image_from_file

Metadata = namedtuple("Metadata", ["target", "camera", "photos"])
Target = namedtuple("Target", ["box", "format"])
Vector = namedtuple("Vector", ["vector", "format"])
Camera = namedtuple("Camera", ["fov", "aspect", "near", "far"])
Image = namedtuple("Image", ["position", "rotation", "image"])

def get_metadata(metaroot): 
    files = os.listdir(metaroot)

    output = {}

    for file in files:
        with open(os.path.join(metaroot, file), "r") as json_file:
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
            photos.append(
                Image(
                    Vector(photo["position"], "XYZ"), 
                    Vector(photo["rotation"][:-1], "XYZ"),
                    photo["image"]
                )
            )

        output[file] = Metadata(target, camera, photos)
        
    return output

def get_image(file):
    return read_image(file)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, meta_root, data_root, transform=None):
        self.metadata = get_metadata(meta_root)
        self.filenames = list(self.metadata)
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        metadata = self.metadata[filename]

        photos = []
        rotations = []
        positions = []

        fov = metadata.camera.fov
        aspect = metadata.camera.aspect
        near = metadata.camera.near
        far = metadata.camera.far

        target = metadata.target.box

        for photo in metadata.photos:
            photos.append(photo.image)
            rotations.append(photo.rotation.vector)
            positions.append(photo.position.vector)
    
        if self.transform:
            photos = [self.transform(load_image_from_file(os.path.join(self.data_root, photo))) for photo in photos]

        return fov, near, far, aspect, photos, positions, rotations, target