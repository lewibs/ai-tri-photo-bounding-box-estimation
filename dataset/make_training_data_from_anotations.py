import os
import shutil
import math
import pandas as pd
import json

def copy_data_for_train(source_dir, destination_dir, anotations):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    train_dir = os.path.join(destination_dir)
    
    os.makedirs(train_dir, exist_ok=True)

    files = []

    df = pd.read_csv(anotations)
    ids = df.values[:, 4]
    metadata = df.values[:, 5]

    for i, data in enumerate(metadata):
        data = json.loads(data)
        basename = os.path.basename(ids[i])
        # Split the string using the '-' delimiter
        basename = basename.split('-')

        # Remove the first element from the resulting list
        basename = basename[1:]

        # Join the remaining parts back together using the '-' delimiter
        basename = '-'.join(basename)

        shutil.copy(os.path.join(source_dir, basename), os.path.join(train_dir, basename))

if __name__ == "__main__":
    source_dir = "./data"
    destination_dir = "./data_train"
    anotations_file = "./anotations.csv"

    try:
        copy_data_for_train(source_dir, destination_dir, anotations_file)
        print("train/validate directories created.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
