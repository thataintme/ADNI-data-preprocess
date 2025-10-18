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
out_dir = 'Preprocessed_Data/MRI_Montage'

# Settings
TOP_DOWN = True  # If True, the montage will be created from top to down. If False, it will be created from down to top.
num_slices_to_extract = 40 # Number of slices to extract from the montage
rows = 5
cols = 8

BLACKNESS_THRESHOLD = 10
PERCENTAGE_THRESHOLD = 10


if num_slices_to_extract % 2 != 0:
    raise ValueError("num_slices and num_slices_to_extract must be even numbers")
if num_slices_to_extract % (rows * cols) != 0:
    raise ValueError("num_slices_to_extract must be divisible by (rows * cols)")


def omit_black_slices(data, axis, blackness_threshold, percentage_threshold):
    """
    Identifies the starting and ending indices along a specified axis in a 3D data array,
    omitting slices that are predominantly "black" based on given thresholds.
    Parameters
    ----------
    data : np.ndarray
        3D array representing the volumetric data to process.
    axis : int
        The axis along which to evaluate slices (0, 1, or 2).
    blackness_threshold : int
        Pixel intensity threshold (0-255). Pixels with values below this are considered "black".
    percentage_threshold : float
        The maximum allowed percentage of "black" pixels in a slice for it to be considered valid.
        Slices with a higher percentage of "black" pixels are omitted.
    normalised_max : int
        The maximum index (exclusive) along the chosen axis to consider.
    Returns
    -------
    start : int or None
        The index of the first slice along the axis that meets the threshold criteria, or None if not found.
    end : int or None
        The index of the last slice along the axis that meets the threshold criteria, or None if not found.
    Notes
    -----
    - "Black" pixels are those with intensity values less than `blackness_threshold`.
    - The function scans from both ends of the axis to find the first and last valid slices.
    - Useful for cropping out empty or irrelevant regions in volumetric data.
    """

    start = -1
    end = -1

    max_index = data.shape[axis] - 1


    for i in range(0,max_index//2):
        if axis == 0:
            slice_2d = data[i, :, :]
        elif axis == 1:
            slice_2d = data[:, i, :]
        else:
            slice_2d = data[:, :, i]
        
        if ((1-(np.sum(slice_2d < blackness_threshold) / slice_2d.size)) * 100 > percentage_threshold):
            start = i
            break

    for j in range(max_index, max_index//2, -1):
        if axis == 0:
            slice_2d = data[j, :, :]
        elif axis == 1:
            slice_2d = data[:, j, :]
        else:
            slice_2d = data[:, :, j]
        
        if ((1-(np.sum(slice_2d < blackness_threshold) / slice_2d.size)) * 100 > percentage_threshold):
            end = j
            break
    
    return [start, end]





def nii_to_montage(nii_path, output_dir, out_file_name, axis=2, num_slices=18, rows=3, cols=4):
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255  # normalize to [0, 255]

    

    # Get 12 evenly spaced slice indices
    [start, end] = omit_black_slices(data, axis, BLACKNESS_THRESHOLD, PERCENTAGE_THRESHOLD)
    indices = np.linspace(start, end, num_slices, dtype=int)

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
        slices.append(Image.fromarray(slice_img))

    w, h = slices[0].size
    montage = Image.new('L', (cols * w, rows * h))
    

    # TODO: Use a lightweight model to choose the best slices, possibly using just the brain region
    # ============================================================================


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
  nii_to_montage(nii_file, out_dir, nii_file_name, num_slices=num_slices_to_extract, rows=rows, cols=cols)
