import pandas as pd
import numpy as np
import os
import csv


# Output path
output_path = '040725-ADNIMERGE_join_HipMask_on_VISCODE.csv'

# Baseline output (optional)
baseline_data_output_path = 'BASELINE-ADNIMERGE_join_HipMask_on_VISCODE.csv'

# Paths to the csv files
adnimerge_path = 'Tabular Data/ADNIMERGE_27Jun2025.csv'
hippocampal_masks_data_path = 'Tabular Data/MRI_Hippocampal_Masks_6_30_2025.csv'
patdemog_path = 'Tabular Data/PTDEMOG_10Jun2025.csv'


# Useful functions
def filter_columns(df, columns):
    return df[columns]

def standardise_date_format(df, date_column, given_format, target_format='%Y-%m-%d'):
    df[date_column] = pd.to_datetime(df[date_column], format=given_format)
    df[date_column] = df[date_column].dt.strftime(target_format)

def filter_rows_by_column_value(df, column, value):
    return df[df[column] == value]

def rename_column(df, old_name, new_name):
    df.rename(columns = {old_name: new_name}, inplace=True)

def rename_columns(df, columns_dict):
    df.rename(columns = columns_dict, inplace=True)


# Step 1:
# Load all csv files
adnimerge = pd.read_csv(adnimerge_path, low_memory=False)
hippocampal_masks_data = pd.read_csv(hippocampal_masks_data_path)
patdemog = pd.read_csv(patdemog_path, low_memory=False)

# Step 2:
# Bring in the image data (which is the ADNIMERGE data) and only keep relevant columns
# Relevant columns: ['PTID', 'RID', 'VISCODE', 'EXAMDATE', 'DX', 'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTMARRY', 'IMAGEUID', 'Hippocampus', 'ICV', 'WholeBrain', 'IMAGEUID_bl', 'Hippocampus_bl', 'ICV_bl', 'WholeBrain_bl']
adnimerge = filter_columns(adnimerge, ['PTID', 'RID', 'VISCODE', 'EXAMDATE', 'DX', 'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTMARRY', 'IMAGEUID', 'Hippocampus', 'ICV', 'WholeBrain', 'IMAGEUID_bl', 'Hippocampus_bl', 'ICV_bl', 'WholeBrain_bl'])
standardise_date_format(adnimerge, 'EXAMDATE', given_format = '%Y-%m-%d')

# Step 3:
# Now we may join the image data with hippocampal masks data
# The columns from hippocampal masks data that are relevant are:
# 'Image Data ID'	'Subject' 'Visit'	'Acq Date'
# Hippocampal mask image IDs have an extra 'I' in the beginning, we will remove it
# One might notice that 'Visit' is the same as VISCODE and we can ignore it to avoid disturbances
# Subject is the same as PTID, and Acq Date is the same as EXAMDATE
# This simply does the job of introducing the 'Image Data ID' (of the hippocampal mask) column to our image data
# We perform the reuired renaming, filtering, and joining
# We do a left join to keep all rows from image_data. This will result in a few extra rows as some items have multiple matches

hippocampal_masks_data = filter_columns(hippocampal_masks_data, ['Image Data ID', 'Subject', 'Visit'])
hippocampal_masks_data['Image Data ID'] = hippocampal_masks_data['Image Data ID'].str.replace('I', '', regex=False)
rename_columns(hippocampal_masks_data, {'Image Data ID': 'HIPPOCAMPAL_MASK_ID', 'Subject': 'PTID', 'Acq Date': 'EXAMDATE', 'Visit': 'VISCODE'})
# standardise_date_format(hippocampal_masks_data, 'EXAMDATE', given_format = '%m/%d/%Y')



# Step 4:
# We are now ready to merge this data with the diagnosis sumary
# This gives us 1.1k rows with hippocampal mask data
# And more importantly, almost all rows contain hippocampal volume and Intracranial volume data
merged_data = pd.merge(adnimerge, hippocampal_masks_data, on=['PTID', 'VISCODE'], how='left')

# Step 5:
# TODO: Try with inner join cuz the left join gives us a lot of extra rows (almost double)
# Now we can add patient demographic data which are not available in the ADNIMERGE data, but present in the PTDEMOG data
# I choose to add the following columns:
# 'PTHAND'
# patdemog = filter_columns(patdemog, ['PTID', 'PTHAND'])
# patdemog['PTHAND'] = patdemog['PTHAND'].map({1: 'Right', 2: 'Left'})

# merged_data = pd.merge(merged_data, patdemog, on='PTID', how='left')



# Step 6:
# Some rows in DX column are missing. These rows have a valid image data but no available Diagnosis.
# Therefore, we use the diagnosis given in a PREVIOUS visit for the same patient.
# This introduces a very slight bias. But is much better than using the baseline diagnosis DX_bl
merged_data = merged_data.sort_values(['PTID', 'EXAMDATE'])
merged_data['DX'] = merged_data.groupby('PTID')['DX'].ffill()

# Step 7:
# Remove rows where either IMAGEUID and DX are null
merged_data = merged_data[merged_data["IMAGEUID"].notnull()]
merged_data = merged_data[merged_data["DX"].notnull()]


# Step 7:
# Finally, we can save the merged data to a new csv file
merged_data.to_csv(output_path, index=False)





# Optional, if time permits, we will be running the fine-tuning on baseline data alone for training with level of AD progression
# This data is only present in the baseline records.
if baseline_data_output_path and len(baseline_data_output_path)>0:
    baseline_data = merged_data[merged_data["VISCODE"] == 'bl']
    baseline_data = baseline_data[['PTID', 'RID', 'VISCODE', 'EXAMDATE', 'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTMARRY', 'IMAGEUID', 'Hippocampus', 'ICV', 'WholeBrain', 'HIPPOCAMPAL_MASK_ID']]
    rename_column(baseline_data, 'DX_bl','DX')
    baseline_data.to_csv(baseline_data_output_path)


# Notes: The field PTNOTRT is defined as "7. Participant Retired?", and this field confuses me heavily.
# It is not clear what it means, and it is not used in the analysis.
