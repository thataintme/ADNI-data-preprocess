import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import argparse
from PIL import Image

# This script converts NiFTi files to a montage of 2D slices.
# It extracts a specified number of slices from the montage
# To ensure that the slices are not predominantly black, I have implemented a function omit_black_slices
# However, this implementation is not perfect and the results seem to corrupt the data, hindering the model's performance.

# parser = argparse.ArgumentParser(description="Convert NiFTi files to montage images.")
# parser.add_argument('--nii_images_path', type=str, required=True, help='Path to input NiFTi images directory')
# parser.add_argument('--out_dir', type=str, required=True, help='Path to output montage images directory')
# args = parser.parse_args()
# nii_images_path = args.nii_images_path
# out_dir = args.out_dir

nii_images_path = 'Raw_Image_Data/TrainingScans'
out_dir = 'Preprocessed_Data/MRI_Montage'

# Settings
TOP_DOWN = True  # If True, the montage will be created from top to down. If False, it will be created from down to top.
SLICES_PER_IMAGE = 20 # Number of slices to extract from the montage
rows = 4
cols = 5
percentage_range = 0.15  # Percentage of the total range of slices to extract from the middle
mid_offset = 0.1 # Offset a little bit lower than the middle to reach the hippocampus region.


if SLICES_PER_IMAGE % 2 != 0:
    raise ValueError("num_slices and num_slices_to_extract must be even numbers")
if SLICES_PER_IMAGE % (rows * cols) != 0:
    raise ValueError("num_slices_to_extract must be divisible by (rows * cols)")




def nii_to_montage(nii_path, output_dir, out_file_name, percentage_range, num_slices, rows, cols, axis=2):
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()

    # data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255  # normalize to [0, 255]
    # Rather than normalising on the whole 3D image, I think it is better to normalise AFTER the segmenting
    # For now I am normalising at the slice level
    

    half_percent = percentage_range / 2
    middle = 0.5 - mid_offset
    total_slices = data.shape[axis]

    middle_start = int(total_slices * (middle - half_percent))
    middle_end = int(total_slices * (middle + half_percent))

    indices = np.linspace(middle_start, middle_end, num_slices, dtype=int)

    if TOP_DOWN:
        indices = indices[::-1]

    slices = []
    for i in indices:
        if axis == 0:
            slice_2d = data[i, :, :]
        elif axis == 1:
            slice_2d = data[:, i, :]
        else:
            slice_2d = data[:, :, i]
        slice_img = (slice_2d).astype(np.uint8)
        # Normalize the slice to [0, 255]
        slice_norm = (slice_2d - np.min(slice_2d))
        if np.max(slice_norm) > 0: # Usually this won't be needed for mid-range, cuz if max is 0, it means it is an empty slice
            slice_norm = slice_norm / np.max(slice_norm)
        slice_img = (slice_norm * 255).astype(np.uint8)
        slices.append(Image.fromarray(slice_img))

    w, h = slices[0].size
    montage = Image.new('L', (cols * w, rows * h))
    

    for idx, slice_img in enumerate(slices):
        row = idx // cols
        col = idx % cols
        montage.paste(slice_img, (col * w, row * h))

    montage.save(os.path.join(output_dir, f"{out_file_name.split('.')[0]}.png"))



# nii_file_name = '101719.nii'
# nii_file = os.path.join(images_path, nii_file_name)
# nii_to_montage(nii_file, out_dir, nii_file_name, num_slices=num_slices_to_extract, rows=rows, cols=cols)


for nii_file_name in tqdm(os.listdir(nii_images_path), desc=f'Prerocessing images'):
  nii_file = os.path.join(nii_images_path, nii_file_name)
  nii_to_montage(nii_file, out_dir, nii_file_name, percentage_range=percentage_range, num_slices=SLICES_PER_IMAGE, rows=rows, cols=cols)
