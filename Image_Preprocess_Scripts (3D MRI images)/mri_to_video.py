import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import cv2


# This script converts NiFTi files to video files.
# It takes every 2D slice along an axis and saves it as a frame in a video file.
# The resulting video can be fed to Video-compatible VLM models for fine-tuning.

# TODO: Consider normalising the size of the images. Some are 256x256, some are 256x166. Current status - not doing it for some level of generalisation
# TODO: Consider using different axes together. Current status - not doing it due to data size constraint


# nii_images_path = 'Raw_Image_Data/TrainingScans'
# out_dir = 'Preprocessed_Data/Video'

nii_images_path = 'Raw_Image_Data/HippocampalMasks/Hippocampal masks'
out_dir = 'Preprocessed_Data/Video_Only_Hip_Mask'

TOP_DOWN = False  # If True, the video will be created from top to down. If False, it will be created from down to top.



def nii_to_video(nii_path, output_dir, out_file_name, axis=2, fps=30):
    video_filename = os.path.join(output_dir, out_file_name)

    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255  # normalize to [0, 255]

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

        if slice_img.max() != 0:
            vid_writer.write(slice_img)
   
    vid_writer.release()


for nii_file_name in tqdm(os.listdir(nii_images_path), desc=f'Prerocessing images'):
  nii_file = os.path.join(nii_images_path, nii_file_name)
  output_video_name = nii_file_name.split('.')[0] + '.avi'
  nii_to_video(nii_file, out_dir, output_video_name)
