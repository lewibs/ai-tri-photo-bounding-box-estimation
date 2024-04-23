import zipfile
import os
import json

# Create directories if they don't exist
if not os.path.exists('metadata'):
    os.makedirs('metadata')
if not os.path.exists('data'):
    os.makedirs('data')

# Open the zip file
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    # Extract json files to metadata folder
    for file_info in zip_ref.infolist():
        if file_info.filename.endswith('.json'):
            zip_ref.extract(file_info, 'metadata')

    # Extract jpg files to data folder
    for file_info in zip_ref.infolist():
        if file_info.filename.endswith('.jpg'):
            zip_ref.extract(file_info, 'data')

# Inform user about completion
print("Extraction and deletion completed successfully.")
