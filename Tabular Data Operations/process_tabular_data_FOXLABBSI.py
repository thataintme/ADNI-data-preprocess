import pandas as pd
import numpy as np
import os
import csv


# Paths to the csv files
image_data_path = 'Tabular Data/FOXLABBSI_30Jun2025.csv'
dxsum_path = 'Tabular Data/DXSUM_30Jun2025.csv' # Diagnosis summary data
hippocampal_masks_data_path = 'Tabular Data/MRI_Hippocampal_Masks_6_30_2025.csv'


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
image_data = pd.read_csv(image_data_path)
dxsum = pd.read_csv(dxsum_path)
hippocampal_masks_data = pd.read_csv(hippocampal_masks_data_path)

# Step 2:
# Bring in the image data and only keep relevant columns
# Relevant columns: PHASE	PTID	RID	VISCODE	VISCODE2	LONIUID	LONIUID_BASE	EXAMDATE	RUNDATE	STATUS	VERSION	BRAINVOL	VENTVOL	HIPPOVOL_R	HIPPOVOL_L
# The columns VENTACCEPT	HPACCEPT_R	HPACCEPT_L	and	QC_PASS indicate whether the data is acceptable
# However, QC_PASS represents "Selected for cross-sectional analysis", which leaves some ambiguity about the quality
# Therefore, we will assume the data is still usable and not use QC_PASS to filter the data in this step
# One can choose to remove rows where HPACCEPT and VENTACCEPT are 0
# In this study, I will keep it as is, to allow for some noise in the data
# Also VISCODE2 is the translated version of VISCODE, and is more consistent with other datasets
# Therefore we will keep it and rename it to VISCODE
image_data = filter_columns(image_data, ['PHASE', 'PTID', 'RID', 'VISCODE2', 'LONIUID', 'LONIUID_BASE', 'EXAMDATE', 'RUNDATE', 'STATUS', 'VERSION', 'BRAINVOL', 'VENTVOL', 'HIPPOVOL_R', 'HIPPOVOL_L'])
rename_column(image_data, 'VISCODE2', 'VISCODE')
standardise_date_format(image_data, 'EXAMDATE', given_format = '%Y-%m-%d')

# Step 3
# Now we may join the image data with hippocampal masks data
# The columns from hippocampal masks data that are relecant are:
# 'Image Data ID'	'Subject' 'Visit'	'Acq Date'
# One might notice that 'Visit' is the same as VISCODE and we can ignore it to avoid disturbances
# Subject is the same as PTID, and Acq Date is the same as EXAMDATE
# This simply does the job of introducing the 'Image Data ID' (of the hippocampal mask) column to our image data
# We perform the reuired renaming, filtering, and joining
# We do a left join to keep all rows from image_data. This will result in a few extra rows as some items have multiple matches
hippocampal_masks_data = filter_columns(hippocampal_masks_data, ['Image Data ID', 'Subject', 'Acq Date'])
rename_columns(hippocampal_masks_data, {'Image Data ID': 'HIPPOCAMPAL_MASK_ID', 'Subject': 'PTID', 'Acq Date': 'EXAMDATE'})
standardise_date_format(hippocampal_masks_data, 'EXAMDATE', given_format = '%m/%d/%Y')



# Step 4:
# We are now ready to merge this data with the diagnosis sumary
merged_data = pd.merge(image_data, hippocampal_masks_data, on=['PTID', 'EXAMDATE'], how='left')
merged_data.to_csv('FOXLABBSI_join_HipMask.csv', index=False)



# Update: This pathway is halted. We are going to use the ADNIMERGE data instead
# Check process_tabular_data_ADNIMERGE.py for the latest code

# This pathway was intended to have both hippocampal masks and hippocampal volume data
# However, the rows which have matching hippocampal masks data do not have hippocampal volume data
# The ADNIMERGE data has hippocampal volume data for almost all rows.


# The output of this code will be saved here. This is because:
#       FOXLABBSI -> MRI_Hippocampal_Masks 
# has 1.5k rows which have matching hippocampal masks data

# ADNIMERGE data only has 1.1k rows with matching hippocampal masks data
# Therefore, we will use the FOXLABBSI data for future analysis.





