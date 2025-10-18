# This will be helpful for downloading only the useful images from the ADNI site

import pandas as pd

# df = pd.read_csv('040725-ADNIMERGE_join_HipMask_on_VISCODE.csv')
# output_comma_separated_txt_path = 'Tabular Data/comma_separated_image_ids.txt'

df = pd.read_csv('BASELINE-ADNIMERGE_join_HipMask_on_VISCODE.csv')
output_comma_separated_txt_path = 'Tabular Data/baseline_comma_separated_image_ids.txt'



df["IMAGEUID"] = df["IMAGEUID"].astype(int).astype(str) # For some reason pandas saves these as floats. So this conversion is needed.
column = df['IMAGEUID']

with open(output_comma_separated_txt_path, "w") as outfile:
    outfile.write(",".join(df['IMAGEUID']))

