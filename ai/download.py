import os

def downloadStuff():
    # URLs of the files to download
    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
    ]

    # Destination directory where the files will be saved
    destination_dir = "../ai"

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Download each file
    for url in urls:
        filename = os.path.join(destination_dir, os.path.basename(url))
        os.system(f"powershell Invoke-WebRequest -Uri {url} -OutFile {filename}")

    # Verify if files are downloaded successfully
    for filename in os.listdir(destination_dir):
        print(filename)

downloadStuff()