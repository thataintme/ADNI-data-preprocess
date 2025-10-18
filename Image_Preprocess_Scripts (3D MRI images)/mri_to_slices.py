import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from PIL import Image

# This script converts NiFTi files to a montage of 2D slices.
# It extracts a specified number of slices from the montage
# To ensure that the slices are not predominantly black, I have implemented a function omit_black_slices
# However, this implementation is not perfect and the results seem to corrupt the data, hindering the model's performance.


nii_images_path = 'Raw_Image_Data/Scans'
out_dir = 'Preprocessed_Data/Slices'

# Settings
TOP_DOWN = True  # If True, the slice indices will be named from top to down. If False, they will be named from down to top.
num_slices_to_extract = 40 # Number of slices to extract from the montage
rows = 5
cols = 8
# start = 0
# end = 0

# Thresholds for slice selection
# These thresholds are used to omit slices that are predominantly black
# BLACKNESS_THRESHOLD decides the pixel intensity below which a pixel is considered "black"
# PERCENTAGE_THRESHOLD decides the minimum percentage of "not black" pixels allowed in a slice.
# 10% means that at least 10% of the pixels in a slice must be above the BLACKNESS_THRESHOLD to be considered valid
BLACKNESS_THRESHOLD = 10
NOT_BLACK_PERCENTAGE_THRESHOLD = 10


if num_slices_to_extract % 2 != 0:
    raise ValueError("num_slices and num_slices_to_extract must be even numbers")
if num_slices_to_extract % (rows * cols) != 0:
    raise ValueError("num_slices_to_extract must be divisible by (rows * cols)")





def nii_to_montage(nii_path, output_dir, out_file_name, num_slices, axis=2):
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255  # normalize to [0, 255]

    

    [start, end] = [0, data.shape[axis] - 1]

    indices = np.linspace(start, end, num_slices, dtype=int)

    for i in indices:
        if axis == 0:
            slice_2d = data[i, :, :]
        elif axis == 1:
            slice_2d = data[:, i, :]
        else:
            slice_2d = data[:, :, i]
        slice_img = Image.fromarray((slice_2d).astype(np.uint8))
        save_path = os.path.join(output_dir, f"{out_file_name.split('.')[0]}_{end-i if TOP_DOWN else i}.png")
        slice_img.save(save_path)

    # TODO: Use a lightweight model to choose the best slices, possibly using just the brain region
    # ============================================================================




# nii_file_name = '101719.nii'
# nii_file = os.path.join(images_path, nii_file_name)
# nii_to_montage(nii_file, out_dir, nii_file_name, num_slices=num_slices_to_extract, rows=rows, cols=cols)


for nii_file_name in tqdm(os.listdir(nii_images_path), desc=f'Prerocessing images'):
  nii_file = os.path.join(nii_images_path, nii_file_name)
  nii_to_montage(nii_file, out_dir, nii_file_name, num_slices=num_slices_to_extract, rows=rows, cols=cols)
