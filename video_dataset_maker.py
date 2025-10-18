from prompt_generator import generate_prompt, generate_hippocampus_description
import pandas as pd
from tqdm import tqdm
import os

# TODO: Different prompt formats to allow for generalisation. After successful finetuning

csv_file = 'AMAS.csv'
videos_dir = 'Preprocessed_Data/Video'
output_csv = 'datasets/video-dataset.csv'

INCLUDE_BASELINE_HIPPOCAMPAL_VOLUME = True

# We will be using more than just the baseline diagnosis, and to keep it simple we condense all levels into just the three main categories:
dx_map = {
    'Dementia': 'AD',
    'MCI': 'MCI',
    'CN': 'CN',
    'AD': 'AD',
    'NL': 'CN',
    'EMCI': 'MCI',
    'LMCI': 'MCI',
    'SMC': 'CN'
}

td = pd.read_csv(csv_file)
td = td[['DX', 'VISCODE', 'PTID', 'IMAGEUID', 'Hippocampus', 'ICV', 'EXAMDATE', 'Hippocampus_bl', 'ICV_bl',
         'PTGENDER', 'AGE', 'PTEDUCAT', 'PTRACCAT', 'PTMARRY']]


df_rows =[]
for vid_fname in tqdm(os.listdir(videos_dir)):
    vid_basename = os.path.splitext(vid_fname)[0]
    rows = td[td['IMAGEUID'] == int(vid_basename)]
    if len(rows) == 0:
        continue
    row = rows.iloc[0]

    vid_path = vid_fname
    label = row.DX
    ptid = row.PTID
    demog_description = generate_prompt(row, prompt_base='', include_hippocampal_volume=False)
    hippocampus_description = generate_hippocampus_description(row, include_baseline_hippocampal_vol_data=INCLUDE_BASELINE_HIPPOCAMPAL_VOLUME)
    df_rows.append({'Video_path': vid_path, 'LABEL': label, 'PTID': ptid, 'DESCRIPTION': demog_description, 'HIPPOCAMPUS_DESC':hippocampus_description})


df = pd.DataFrame(df_rows)

df.to_csv(output_csv, index=False)



# In case you want to create the whole prompt (demographic and hippocampal descriptions) in one go, use the following lines.
# Do not use newlines in prompt for llava-med as it uses JSONL files for questions.
# prompt_base = (
#             f"Based on this MRI, what do you think this patient has? Respond with just one of the following: 'Alzheimer's disease', 'Mild cognitive impairment', or 'Normal control'. No extra text. "
#         )
