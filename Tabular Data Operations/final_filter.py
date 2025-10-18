# After querying the images using the comma-separated IMAGEUIDs on ADNI, it was noted that
# out of 10236 images present in the tabular dataset, only 8188 are available in the ADNI dataset

# It is possible that some data was lost, or consent revoked, or quality concerns were found at later points of time.
# The finally available in 'Tabular Data/Phase-4_Final_Images_7_04_2025.csv'
# This data will be used to filter our newly created final dataset

# It was also noted that some scans have more than one hippocampal mask images associated with them. These are 240 in number
# This means there are 240 duplicates. As this is a relatively small number, I see no harm in keeping these values in.

# In this step, I have also decided to name this dataset as "AMAS" set, for ADNI Merge Available Set
# AMAS has 8309 rows

import pandas as pd

output_final_csv = 'AMAS.csv'
output_final_csv_baseline = 'AMAS-baseline.csv'

adni_merge_hip_csv_path = 'Tabular Data/040725-ADNIMERGE_join_HipMask_on_VISCODE.csv'
baseline_hip_csv_path = 'Tabular Data/BASELINE-ADNIMERGE_join_HipMask_on_VISCODE.csv'
available_images_csv_path = 'Tabular Data/Phase-4_Final_Images_7_04_2025.csv'


adni_merge_hip_csv = pd.read_csv(adni_merge_hip_csv_path)
baseline_hip_csv = pd.read_csv(baseline_hip_csv_path)
available_images_csv = pd.read_csv(available_images_csv_path)


available_images_csv['IMAGEUID'] = available_images_csv['Image Data ID'].str.replace('I', '', regex=False)
adni_merge_hip_csv['IMAGEUID'] = adni_merge_hip_csv['IMAGEUID'].astype(int).astype(str)
baseline_hip_csv['IMAGEUID'] = baseline_hip_csv['IMAGEUID'].astype(int).astype(str)


adni_merge_hip_filtered = adni_merge_hip_csv[adni_merge_hip_csv['IMAGEUID'].isin(available_images_csv['IMAGEUID'])]
baseline_hip_filtered = baseline_hip_csv[baseline_hip_csv['IMAGEUID'].isin(available_images_csv['IMAGEUID'])]


adni_merge_hip_filtered.to_csv(output_final_csv)
baseline_hip_filtered.to_csv(output_final_csv_baseline)
