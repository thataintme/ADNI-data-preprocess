# ADNI-data-preprocess
This repository contains the code developed during my research project on Alzheimer’s disease detection using neuroimaging. The study utilized the ADNI dataset which is the most abundant dataset available for Alzheimer’s research.

The project explored the utility of image-based machine learning techniques to detect Alzheimer's disease from MRI scans. This was deeper into the utility of Vision-Language models, most of which only accept 2D images as input data as of 2025. The code here is used to convert NIfTI formatted (.nii) 3D MRI images to a 2D format that the VLMs can process. This was done by extracting 2D slices along an axis, creating montages in an attempt to maintain physical structural features, etc. Additionally, there's scripts that use tabular data to perform operations such as adding a "yearly hippocampal atrophy rate" to the data.

The repository has several self-documented scripts that process data from the ADNI dataset and produce outputs as needed. Required tables and images for each process are shown in each .py file.

⚠️ Note: This repository is research-oriented and not structured. The code is quite messy, as it was written during the study phase and not for software development.

There is an outline of how I attempted adding hippocampal masks in an attempt to direct attention maps of the models I used, but I did not happen to use it in our study. Hopefully it helps someone in the future who takes this direction ^.^
