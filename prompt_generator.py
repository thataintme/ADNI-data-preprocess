import csv
import os
import json
import math

output_dataset_path = './dataset.json'

# prompt_base = (
#             f"Based on this MRI, what do you think this patient has? "
#             f"Respond with just one of the following: 'Alzheimer's disease', 'Mild cognitive impairment', or 'Normal control'. No extra text.\n"
#         )

def generate_prompt(row, prompt_base, include_hippocampal_volume = True):
    """    Generates a prompt for each image based on the patient details.
    Args:
        row (dict): A dictionary containing patient details. This should be a row from the CSV file that contains patient demographics.
    The CSV file should contain columns like PTID, PTGENDER, PTDOB, PTHAND, PTMARRY, PTEDUCAT, PTADBEG, PTCOGBEG, PTMCIBEG, PTETHCAT, PTRACCAT.
        prompt_base (str): Base prompt string to which patient details will be appended.
    Returns:
        str: A prompt string for the model.
    """

    # Got the row, now create a prompt
    prompt = "The subject is "

    if row.get('PTGENDER'):
        gender = row.get('PTGENDER')
        prompt += f"{gender}, "

    if row.get('PTRACCAT'):
        race_cat = row.get('PTRACCAT')
        prompt += f" race category - {race_cat}, "

    if row.get('AGE'):
        age = row.get('AGE')
        if age is not None:
            prompt += f"aged {age} years, "
        else:
            prompt += "age unknown, "
    

    if row.get('PTEDUCAT'):
        prompt += f'has an education level of {row.get("PTEDUCAT")} out of 20, '

    if row.get('PTMARRY'):
        prompt += f'{row.get("PTMARRY").lower()}, '


    # End the sentence.
    if prompt.endswith(', '):
        prompt = prompt[:-2] + '. '

    if (include_hippocampal_volume):
        # ALERT:
        # Hippocampus and ICV data should be something that the model should estimate to make a decision.
        # However, in this study we are assessing the model's ability to provide explainable results.
        # This script is only to make it easy for the model to identfy whether to answer with "AD" or "MCI" or "NC"
        # In later stages of this study, I started asking the model to estimate the atrophy itself from the image

        if row.get('Hippocampus') and row.get('ICV'):
            hippo = row.get('Hippocampus')
            icv = row.get('ICV')
            if hippo is not None and icv is not None and hippo != 0 and icv != 0:
                ratio = (hippo / icv) * 100
                prompt += f'In this scan, patient\'s hippocampus occupies {ratio:.4f}% of the ICV, '


        if row.get('Hippocampus_bl') and row.get('ICV_bl'):
            hippo_bl = row.get('Hippocampus_bl')
            icv_bl = row.get('ICV_bl')
            if hippo_bl is not None and icv_bl is not None:
                ratio_bl = (hippo_bl / icv_bl) * 100
                prompt += f'and on their first visit, this percentage was {ratio_bl:.4f}%, '

        # End the sentence
        if prompt.endswith(', '):
            prompt = prompt[:-2] + '. '
    
    return prompt_base + prompt



def generate_hippocampus_description(row, include_baseline_hippocampal_vol_data = True):
    hippo_desc = ''
    cur_hippo_data_present = False
    if row.get('Hippocampus') and row.get('ICV'):
        hippo = row.get('Hippocampus')
        icv = row.get('ICV')
        if hippo is not None and icv is not None and (not math.isnan(hippo)) and (not math.isnan(icv)) and hippo != 0 and icv != 0:
            ratio = (hippo / icv) * 100
            hippo_desc += f'In this scan, patient\'s hippocampus occupies {ratio:.4f}% of the ICV; '
            cur_hippo_data_present = True


    viscode = row.get("VISCODE")
    if include_baseline_hippocampal_vol_data and (viscode.lower() != 'bl') and row.get('Hippocampus_bl') and row.get('ICV_bl'):
        hippo_bl = row.get('Hippocampus_bl')
        icv_bl = row.get('ICV_bl')
        n_months_from_bl = viscode.replace('m', '')
        if hippo_bl is not None and icv_bl is not None and (not math.isnan(hippo_bl)) and (not math.isnan(icv_bl)) and hippo_bl != 0 and icv_bl != 0:
            ratio_bl = (hippo_bl / icv_bl) * 100
            if cur_hippo_data_present:
                hippo_desc += f'{n_months_from_bl} months ago, this percentage was {ratio_bl:.4f}%, '
            else:
                hippo_desc += f'Hippocampal volume data for this scan is unavailable, but {n_months_from_bl} months ago, hippocampus occupies {ratio_bl:.4f}% of the ICV, '
    
    # End the sentence
    if hippo_desc.endswith(', ') or hippo_desc.endswith('; '):
        hippo_desc = hippo_desc[:-2] + '. '
    
    return hippo_desc




"""
Just a comment for reference. These are the fields that are available in the merged_output.csv
Available fields in the merged CSV file:

THESE CAN BE USED FOR PROMPT-GENERATION:
    - PTID
    - PTGENDER
    - PTDOB
    - PTHAND
    - PTMARRY
    - PTEDUCAT
    - PTADBEG
    - PTCOGBEG
    - PTMCIBEG
    - PTETHCAT
    - PTRACCAT

THESE CAN ONLY BE USED FOR FEW-SHOT AND FINE-TUNING:
"""