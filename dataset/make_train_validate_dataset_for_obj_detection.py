import os
import shutil
import math

def copy_data_for_train(source_dir, destination_dir, split):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    train_dir = os.path.join(destination_dir, "train")
    val_dir = os.path.join(destination_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List all files in the source directory
    files = os.listdir(source_dir)
    
    # Calculate the number of files needed for training
    split_count = math.ceil(len(files) * split)
    
    # Create folders and copy images
    for i in range(split_count):
        file = files[i]
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

    for i in range(split_count, len(files)):
        file = files[i]
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

if __name__ == "__main__":
    source_dir = "./data"
    destination_dir = "./data_train"
    train_split = 0.9  # 90% of data for training, 10% for validation

    try:
        copy_data_for_train(source_dir, destination_dir, train_split)
        print("train/validate directories created.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
