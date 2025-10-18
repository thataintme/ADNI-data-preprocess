# Due to storage constraints, I was only using a subset of all available images in the ADNI dataset
# This script is meant to help me filter out scans that won't be useful for this study, and to get a list
# of images that I need to download (so that I don't re-download images I have in my system already)

import os
import pandas as pd

folder_path = 'Raw_Image_Data/Scans'

filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

amas = pd.read_csv('AMAS.csv')

image_ids = amas['IMAGEUID'].astype(str).to_list()
missing_image_ids = [img_id for img_id in image_ids if img_id not in filenames]

with open('Tabular Data/pending_download_imageids.txt', 'w') as f:
    f.write(','.join(missing_image_ids))


extra_images = [file for file in filenames if file not in image_ids]
for img_id in extra_images:
    current_path = 'Raw_Image_Data/Scans' + '/' + img_id+ '.nii'
    dest_path = 'Raw_Image_Data/UNAVAILABLE_DATA' + '/' + img_id + '.nii'
    os.replace(current_path, dest_path)

