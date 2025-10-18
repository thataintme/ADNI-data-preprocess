
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

amas = pd.read_csv('AMAS.csv')

amas['years'] = amas['VISCODE'].apply(lambda x: 0 if x == 'bl' else int(x[1:]) / 12)

amas['hip_per_icv'] = 100 * amas['Hippocampus'] / amas['ICV']
amas['hip_per_icv_bl'] = 100 * amas['Hippocampus_bl'] / amas['ICV_bl']

valid_ptids = amas['PTID'].dropna().unique()
amas_valid = amas[amas['PTID'].isin(valid_ptids)]
plt.figure(figsize=(12, 6))
# Sample subset to avoid clutter
sample_ptids = valid_ptids[:100]  # limit to 100 patients for clarity

for ptid in sample_ptids:
    patient = amas_valid[amas_valid['PTID'] == ptid].sort_values('years')
    if len(patient) > 1:
        plt.plot(patient['years'], patient['hip_per_icv'], alpha=0.4)

plt.title("hip_per_icv over Time for Each Patient (Starting from Baseline)")
plt.xlabel("Years Since Baseline")
plt.ylabel("Hippocampal Volume / ICV")
plt.grid(True)
plt.show()

# We can see that during the first year (visits m03 and m06) there is severe noise that can affect the model's understanding
# removing the first year
amas = amas[~amas['years'].isin([0.25, 0.5])]
valid_ptids = amas['PTID'].dropna().unique()
amas_valid = amas[amas['PTID'].isin(valid_ptids)]
plt.figure(figsize=(12, 6))
sample_ptids = valid_ptids[:100]  # limit to 100 patients for clarity
for ptid in sample_ptids:
    patient = amas_valid[amas_valid['PTID'] == ptid].sort_values('years')
    if len(patient) > 1:
        plt.plot(patient['years'], patient['hip_per_icv'], alpha=0.4)

plt.title("hip_per_icv over Time for Each Patient (Starting from Baseline)")
plt.xlabel("Years Since Baseline")
plt.ylabel("Hippocampal Volume / ICV")
plt.grid(True)
plt.show()


print()
