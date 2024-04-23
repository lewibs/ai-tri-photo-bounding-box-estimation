import os
import shutil
import math

def copy_data_segmented(source_dir, destination_dir, num_images_per_folder):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # List all files in the source directory
    files = os.listdir(source_dir)
    
    # Calculate the number of folders needed
    num_folders = math.ceil(len(files) / num_images_per_folder)
    
    # Create folders and copy images
    for i in range(num_folders):
        folder_path = os.path.join(destination_dir, str(i))
        os.makedirs(folder_path)
        
        start_idx = i * num_images_per_folder
        end_idx = min((i + 1) * num_images_per_folder, len(files))
        
        for j in range(start_idx, end_idx):
            file = files[j]
            shutil.copy(os.path.join(source_dir, file), folder_path)

if __name__ == "__main__":
    source_dir = "./data"
    destination_dir = "./data_segmented"
    num_images_per_folder = 30

    copy_data_segmented(source_dir, destination_dir, num_images_per_folder)
    print("Data segmented successfully.")
