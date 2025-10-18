import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from PIL import Image
from scipy.ndimage import zoom

# The hippocampus is a very important region that is to be studied for Alzheimer's disease.
# It is hidden in the middle of the brain, so it is important to extract slices from the middle.
# This script uses a defined percentage range to extract slices from the middle of the montage.


nii_images_path = 'Raw_Image_Data/Scans'
out_dir = 'Preprocessed_Data/Mid_slices_1mm'
adnimerge_path = 'AMAS.csv'


# Settings
TOP_DOWN = True  # If True, the slice indices will be named from top to down. If False, they will be named from down to top.
SLICES_PER_IMAGE = 2
percentage_range = 0.05  # Percentage of the total range of slices to extract from the middle
mid_offset = 0.1 # Offset a little bit lower than the middle to reach the hippocampus region.


def nii_to_slices_percentage_range(nii_path, output_dir, out_file_name, percentage_slices, axis=2):
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()


    orig_spacing = img.header.get_zooms()
    zoom_factors = [orig_spacing[i] / 1.0 for i in range(3)]
    data = zoom(data, zoom=zoom_factors, order=1)


    half_percent = percentage_slices / 2
    middle = 0.5 - mid_offset
    total_slices = data.shape[axis]

    middle_start = int(total_slices * (middle - half_percent))
    middle_end = int(total_slices * (middle + half_percent))

    # indices = np.linspace(middle_start, middle_end, SLICES_PER_IMAGE, dtype=int) # Equidistant slices
    indices = np.random.choice(range(middle_start, middle_end), SLICES_PER_IMAGE, replace=False) # Randomly picked slices



    for i in indices:
        if axis == 0:
            slice_2d = data[i, :, :]
        elif axis == 1:
            slice_2d = data[:, i, :]
        else:
            slice_2d = data[:, :, i]
        slice_2d_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8) * 255
        slice_img = Image.fromarray(slice_2d_norm.astype(np.uint8))
        save_path = os.path.join(output_dir, f"{out_file_name.split('.')[0]}_{i}.png")
        slice_img.save(save_path)

for nii_file_name in tqdm(os.listdir(nii_images_path), desc=f'Preprocessing images'):
  nii_file = os.path.join(nii_images_path, nii_file_name)
  nii_to_slices_percentage_range(nii_file, out_dir, nii_file_name, percentage_slices=percentage_range)
