import os
import json
import random
import string

# Directory containing the files
directory = "./backgrounds"

# Get the list of files in the directory
files = os.listdir(directory)

# List to store photo names
photo_names = []

# Iterate through the files and rename them
for i, filename in enumerate(files):
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(directory, filename)):
        # Create the new filename
        new_filename = random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)) + ".jpg"
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

files = os.listdir(directory)

# Iterate through the files and rename them
for i, filename in enumerate(files):
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(directory, filename)):
        # Create the new filename
        new_filename = str(i) + ".jpg"
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        # Append the new filename to the list
        photo_names.append(new_filename)

# Count of photos
count = len(photo_names)

# Create dictionary for JSON
meta_data = {
    "photo_names": photo_names,
    "photo_count": count
}

# Directory containing the files
directory = "./furniture"

# Get the list of files in the directory
files = os.listdir(directory)

# List to store photo names
photo_names = []

# Iterate through the files and rename them
for i, filename in enumerate(files):
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(directory, filename)):
        # Create the new filename
        new_filename = random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)) + ".glb"
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

files = os.listdir(directory)

# Iterate through the files and rename them
for i, filename in enumerate(files):
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(directory, filename)):
        # Create the new filename
        new_filename = str(i) + ".glb"
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        # Append the new filename to the list
        photo_names.append(new_filename)

# Count of photos
count = len(photo_names)

# Create dictionary for JSON
meta_data["furniture_names"] = photo_names
meta_data["furniture_count"] = count

# Write JSON data to file
with open("meta.json", "w") as json_file:
    json.dump(meta_data, json_file)