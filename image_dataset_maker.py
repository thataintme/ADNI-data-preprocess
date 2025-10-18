from prompt_generator import generate_prompt, generate_hippocampus_description
import pandas as pd
from tqdm import tqdm
import os

# TODO: Different prompt formats to allow for generalisation. After successful finetuning

csv_file = 'AMAS.csv'
slices_dir = 'Preprocessed_Data/Slices_2_random_mid'
# output_csv = 'Training_Files/Med-Llava/Image_Slice_Description_Label_no_hippo_data.csv'
output_csv = 'Inference_Files/Phi-4-multimodal-instruct/image-dataset.csv'

INCLUDE_BASELINE_HIPPOCAMPAL_VOLUME = True

# We will be using more than just the baseline diagnosis, and to keep it simple we condense all levels into just the three main categories:
dx_map = {
    'Dementia': 'AD',
    'AD': 'AD',
    'MCI': 'MCI',
    'EMCI': 'MCI',
    'LMCI': 'MCI',
    'NC': 'NC',
    'CN': 'NC',
    'NL': 'NC',
    'SMC': 'NC'
}

td = pd.read_csv(csv_file)
td = td[['DX', 'VISCODE', 'PTID', 'IMAGEUID', 'Hippocampus', 'ICV', 'EXAMDATE', 'Hippocampus_bl', 'ICV_bl',
         'PTGENDER', 'AGE', 'PTEDUCAT', 'PTRACCAT', 'PTMARRY']]


df_rows =[]
for img_fname in tqdm(os.listdir(slices_dir)):
    img_id = img_fname.split('_')[0]

    rows = td[td['IMAGEUID'] == int(img_id)]
    if len(rows) == 0:
        continue
    row = rows.iloc[0]

    img_path = img_fname
    label = dx_map[row.DX]
    dx = row.DX
    ptid = row.PTID
    age = row.AGE
    viscode = row.VISCODE
    hip_vol_percentage = 100*(float(row.Hippocampus) / float(row.ICV)) if row.ICV != 0 else None
    hip_vol_percentage_bl = 100*(float(row.Hippocampus_bl) / float(row.ICV_bl)) if row.ICV != 0 else None
    demog_description = generate_prompt(row, prompt_base='', include_hippocampal_volume=False)
    hippocampus_description = generate_hippocampus_description(row, include_baseline_hippocampal_vol_data=INCLUDE_BASELINE_HIPPOCAMPAL_VOLUME)
    df_rows.append({'Image_path': img_path, 'DX':dx,
                    'LABEL': label, 'PTID': ptid,
                    'VISCODE': viscode, 'AGE': age, 'PTEDUCAT': row.PTEDUCAT,
                    'PTMARRY': row.PTMARRY,
                    'hip_vol_percentage': hip_vol_percentage, 'hip_vol_percentage_bl': hip_vol_percentage_bl,
                    'DESCRIPTION': demog_description, 'HIPPOCAMPUS_DESC':hippocampus_description})


df = pd.DataFrame(df_rows)

df.to_csv(output_csv, index=False)

