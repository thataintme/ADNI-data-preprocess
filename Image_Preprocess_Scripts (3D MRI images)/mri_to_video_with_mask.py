import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import cv2
from PIL import Image, ImageColor, ImageDraw, ImageEnhance
from scipy.ndimage import binary_dilation, gaussian_filter

# This script converts NiFTi files to video files.
# It takes every 2D slice along an axis and saves it as a frame in a video file.
# The resulting video can be fed to Video-compatible VLM models for fine-tuning.

# TODO: Consider normalising the size of the images. Some are 256x256, some are 256x166
# TODO: Consider using different axes together


nii_images_path = 'Raw_Image_Data/Scans'
masks_path = 'Raw_Image_Data/Renamed_HippocampalMasks'  # Results after running rename_hippocampal_mask_filenames.py ; now masks and images have the same names
out_dir = 'Preprocessed_Data/Video_with_Masks'

TOP_DOWN = False  # If True, the video will be created from top to down. If False, it will be created from down to top.


def nii_to_video(nii_path, mask_path, output_dir, out_file_name, axis=2, fps=30):
    video_filename = os.path.join(output_dir, out_file_name)

    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)

    mask = nib.load(mask_path)
    mask = nib.as_closest_canonical(mask)

    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255  # normalize to [0, 255]

    mask_data = mask.get_fdata()
    mask_data = (mask_data != 0).astype(np.uint8) * 255

    # mask_data = (mask_data - np.min(mask_data)) / (np.max(mask_data) - np.min(mask_data)) * 255  # normalize to [0, 255]


    # Figure out w and h
    # Important note: .shape gives h,w
    #  cv2.VideoWriter expected (w,h)
    h, w = None, None
    if axis == 0:
        h, w = data[0, :, :].shape
    elif axis == 1:
        h, w = data[:, 0, :].shape
    else:
        h, w = data[:, :, 0].shape
    

    vid_writer = cv2.VideoWriter(filename=video_filename, fourcc=0, fps=fps, frameSize=(w, h))

    if TOP_DOWN:
        r = range(data.shape[axis] - 1, -1, -1)
    else:
        r = range(data.shape[axis])

    for i in r:
        if axis == 0:
            slice_2d = data[i, :, :]
        elif axis == 1:
            slice_2d = data[:, i, :]
        else:
            slice_2d = data[:, :, i]
        slice_img = (slice_2d).astype(np.uint8)
        slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel

        mask_image = mask_data[:,:,i]

        # if mask_image.max() == 0:
        #     # If the mask is empty, just add the slice without mask
        #     vid_writer.write(slice_img)
        # else:
        #     mask_image = (mask_image).astype(np.uint8)
        #     mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        #     # highlighted_slice = add_highlight(slice_img, mask_image)
        #     # vid_writer.write(highlighted_slice)
       
        if mask_image.max() != 0:
            mask_image = (mask_image).astype(np.uint8)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            masked_slice = mask_only_focus_region(slice_img=slice_img,mask_image=mask_image)
            vid_writer.write(masked_slice)
   
    vid_writer.release()


# This function was written by ChatGPT, made modifications to make it work
def add_highlight(slice_img, mask_image, expand_kernel_size=7, outline_blur=3):
    # Inputs: slice_img (uint8, HxWx3), mask_image (uint8, HxWx3)
    mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # 1. Ensure binary mask
    _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    # 2. Expand the mask outward (this is your focus region)
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_kernel_size, expand_kernel_size))  # larger for more expansion
    expanded_mask = cv2.dilate(mask_bin, expand_kernel, iterations=2)
    # 3. Get just the outer boundary of the expanded region
    outline = cv2.morphologyEx(expanded_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # 4. Blur it to make it fuzzy/soft
    fuzzy_outline = cv2.GaussianBlur(outline, (outline_blur, outline_blur), sigmaX=2)
    # 5. Create a blue outline image
    blue_outline = np.zeros_like(slice_img)
    blue_outline[..., 0] = fuzzy_outline  # blue channel (BGR)
    # 6. Overlay the outline onto the original slice
    overlay = cv2.addWeighted(slice_img, 1.0, blue_outline, 0.7, 0)
    return overlay

def mask_only_focus_region(slice_img, mask_image):
    # Inputs: slice_img (uint8, HxWx3), mask_image (uint8, HxWx3)
    mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # Convert to binary mask
    _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    # Make sure mask has 3 channels
    mask_3ch = cv2.merge([mask_bin] * 3)
    # Apply the mask: keep only the region where mask is 255
    masked_output = cv2.bitwise_and(slice_img, mask_3ch)
    return masked_output


for mask_file_name in tqdm(os.listdir(masks_path), desc=f'Mask nii images'):
    mask_file = os.path.join(masks_path, mask_file_name)
    # Check if the image file with the same name exists
    nii_file = os.path.join(nii_images_path, mask_file_name)

    if not os.path.exists(nii_file):
        # print(f"Corresponding nii file {nii_file} does not exist for mask {mask_file}. Skipping.")
        continue

    output_video_name = mask_file_name.split('.')[0] + '.avi'
    nii_to_video(nii_file, mask_file, out_dir, output_video_name)
