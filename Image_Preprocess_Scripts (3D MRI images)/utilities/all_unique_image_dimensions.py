# Due to varying machine manufacturers and configurations, the obtained images have different dimensions.
# This script will find all unique image dimensions in the given directory.
# This information can be useful to define a standard for each size of the image.

# For example, if the images are 256x256, 256x166
# we determine that on average the slices of the brain in these images are at range 111 to 150 for 256x256 images
# and 90 to 110 for 256x166 images.

# This can be used for lightweight classification training of a model to detect Alzheimer's disease.
import os
import nibabel as nib
from tqdm import tqdm

image_path = 'Raw_Image_Data/Scans'  # Path to the directory containing the images

unique_shapes = set()

for i in tqdm(os.listdir(image_path)):
    if not i.endswith('.nii'):
        continue
    image_file = os.path.join(image_path, i)
    if not os.path.exists(image_file):
        continue
    img = nib.load(image_file)
    img = nib.as_closest_canonical(img)
    datashape = img.get_fdata().shape
    unique_shapes.add(datashape)


print("Unique dimensions: ", unique_shapes)

# Usually, {(160, 192, 192), (160, 256, 240), (166, 256, 256)}