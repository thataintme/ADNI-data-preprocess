# In this file, we rename hippocampal mask files to match the MRI .nii files.
# This makes it easier to pair the masks with the corresponding MRI scans for later stages of processing.

# Ideally this script is supposed to handle the case of multiple scans having the same HIPPOCAMPAL_MASK_ID
# But in the current dataset, each HIPPOCAMPAL_MASK_ID is unique, so I am not implementing that logic.

import pandas as pd
import os
import shutil

dataset_csv_path = 'Tabular Data/ADNIMERGE_join_HipMask_on_VISCODE.csv' # Ensure correct name. This is the output of the script process_tabular_data_ADNIMERGE.py

source_folder = 'Raw_Image_Data/HippocampalMasks'  # Folder where the hippocampal mask files are stored
destination_folder = 'Raw_Image_Data/Renamed_HippocampalMasks'  # Folder where you want them to go to

COPY = False # Set to True if you want to copy files instead of moving them

csv_reader = pd.read_csv(dataset_csv_path, dtype=object, keep_default_na=False)

for idx, row in csv_reader.iterrows():
    mask_id = row.get('HIPPOCAMPAL_MASK_ID')
    if not (pd.notna(mask_id) and str(mask_id).strip() != ''):
        continue

    image_uid = row.get('IMAGEUID')
    if not (pd.notna(image_uid) and str(image_uid).strip() != ''):
        continue
    image_uid = int(float(row.get('IMAGEUID')))

    source_file = os.path.join(source_folder, f"{mask_id}.nii")
    if not os.path.exists(source_file):
        # print(f"Source file {source_file} does not exist. Skipping.")
        continue
    destination_file = os.path.join(destination_folder, f"{image_uid}.nii")
    if COPY:
        shutil.copy(source_file, destination_file)
    else:
        shutil.move(source_file, destination_file)
    
