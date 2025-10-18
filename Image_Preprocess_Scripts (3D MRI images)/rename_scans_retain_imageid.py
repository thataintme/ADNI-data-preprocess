import os

folder_path = 'Raw_Image_Data/TrainingScans'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        parts = filename.split('_')
        new_name = parts[-1].replace('I','')
        new_path = os.path.join(folder_path, new_name)
        os.rename(file_path, new_path)