from PIL import Image
from torchvision import tv_tensors

# Define a function to load an image from a local file path as a PIL image
def load_image_from_file(file_path):
    # Open the image using PIL
    image = Image.open(file_path)
    
    return tv_tensors.Image(image)
