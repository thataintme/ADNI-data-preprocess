import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


amas = pd.read_csv('AMAS.csv')

amas_nobl = amas[amas['VISCODE'] != 'bl'].copy()
amas_nobl['years'] = amas_nobl['VISCODE'].str[1:].astype(int) / 12

amas_nobl['hip_per_icv'] = 100 * amas_nobl['Hippocampus'] / amas_nobl['ICV']
amas_nobl['hip_per_icv_bl'] = 100 * amas_nobl['Hippocampus_bl'] / amas_nobl['ICV_bl']

amas_nobl['atrophied_hip_per_icv'] = (amas_nobl['hip_per_icv_bl'] - amas_nobl['hip_per_icv'])
amas_nobl['atrophy_percent'] = 100 * (amas_nobl['atrophied_hip_per_icv']) / amas_nobl['hip_per_icv_bl']
amas_nobl['atrophy_per_year'] = amas_nobl['atrophy_percent'] / amas_nobl['years']
amas_nobl = amas_nobl[amas_nobl['atrophy_per_year'].notna()]

# Add data on whether the patient is a converter.
# i.e., the patient descends into dementia
def is_progressive_converter(df):
    # Sort by time
    dx_seq = df.sort_values("years")["DX"].tolist()

    # Check for progressive patterns only
    # Don't count people who recover (e.g., MCI -> NC)
    if "CN" in dx_seq:
        cn_index = dx_seq.index("CN")
        later_dxs = dx_seq[cn_index + 1:]
        if any(d in ["MCI", "AD"] for d in later_dxs):
            return True

    if "MCI" in dx_seq:
        mci_index = dx_seq.index("MCI")
        later_dxs = dx_seq[mci_index + 1:]
        if "AD" in later_dxs:
            return True

    return False
converters = amas_nobl.groupby("PTID").apply(is_progressive_converter).reset_index()
converters.columns = ["PTID", "converter"]
amas_nobl = amas_nobl.merge(converters, on="PTID", how="left")


# Boxplot of hip_per_icv of each diagnosis class
sns.boxplot(x='DX', y='hip_per_icv', data=amas_nobl)
plt.show()
# Clearly there are some outliers for both CN and Dementia patients. With respect to clinical knowledge that Cognitively normal patients
# have higher hippocampal volume per ICV than Demented patients, we only remove the outliers from CN beyond the lower whisker.
def remove_iqr_outliers(df, group_col, target_col):
    cleaned_df = pd.DataFrame()
    for _, group_df in df.groupby(group_col):
        Q1 = group_df[target_col].quantile(0.25)
        Q3 = group_df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR
        filtered = group_df[
            (group_df[target_col] >= lower_whisker) &
            (group_df[target_col] <= upper_whisker)
        ]
        cleaned_df = pd.concat([cleaned_df, filtered], axis=0)
    return cleaned_df
# Apply
amas_nobl = remove_iqr_outliers(amas_nobl, group_col='DX', target_col='hip_per_icv')


# Plot Age vs hippocampus_vol_per_ICV
plt.figure(figsize=(8, 6))
plt.scatter(amas_nobl['AGE'], amas_nobl['hip_per_icv'], alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Hippocampus / ICV')
plt.title('Relationship between Age and Hippocampal Volume (normalized)')
plt.grid(True)
plt.show()
# We see that there is one outlier with a 0.9% hippocampal volume in the ICV. Removing that
amas_nobl = amas_nobl[amas_nobl['hip_per_icv'] < 0.8]


# Now plot the hippocampal atrophy percentage against number of years passed
plt.figure(figsize=(8, 5))
plt.scatter(amas_nobl['years'], amas_nobl['atrophy_percent'], color='blue', alpha=0.6)
plt.title('Atrophy percentage vs Years passed')
plt.xlabel('Years passed')
plt.ylabel('Atrophy percentage')
plt.grid(True)
plt.show()
# There are a few points that may be considered outliers.
# But more importantly, it is evident that the entries from the 9th year break a pattern.
# We might choose to remove entries from the 9th year, however I would leave them in as they are small in number
amas_nobl = amas_nobl[
    # (amas_nobl['years'] != 9) & 
    (amas_nobl['atrophy_percent'] > -20) &
    (amas_nobl['atrophy_percent'] < 40)
]


# ICV vs WholeBrain. These should be correlated and if there was an issue here, it could affect our other data
# plt.figure(figsize=(8, 5))
# plt.scatter(amas_nobl['WholeBrain'], amas_nobl['ICV'], color='blue', alpha=0.6)
# plt.grid(True)
# plt.show()
# No obvious issue in the data



# Now plot the ANNUAL hippocampal atrophy percentage against Age
plt.figure(figsize=(8, 5))
plt.scatter(amas_nobl['AGE'], amas_nobl['atrophy_per_year'], color='blue', alpha=0.6)
plt.title('Atrophy per Year vs Age')
plt.xlabel('Age')
plt.ylabel('Atrophy per Year')
plt.grid(True)
plt.show()



# GPT-Enhanced code for plot
# Plot the Atrophied hippocampal ratio against the number of years that passed since baseline visit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Define a color map for DX groups
color_map = {
    'CN': 'green',     # Cognitively Normal
    'MCI': 'orange',   # Mild Cognitive Impairment
    'Dementia': 'red'        # Alzheimer's Disease
}
colors = amas_nobl['DX'].map(color_map).fillna('gray').tolist()
plt.figure(figsize=(8, 5))
plt.scatter(amas_nobl['atrophied_hip_per_icv'], amas_nobl['years'], 
            c=colors, alpha=0.6)
plt.title('Atrophied Hippocampus-ICV Ratio vs Years')
plt.xlabel('Atrophied Hippocampal Volume / ICV')
plt.ylabel('Years Since Baseline')
plt.grid(True)
legend_elements = [Line2D([0], [0], marker='o', color='w', label=dx,
                          markerfacecolor=col, markersize=8)
                   for dx, col in color_map.items()]
plt.legend(handles=legend_elements, title='Diagnosis')
plt.show()
# We can see a clear curve pattern, and some outliers. We can remove the outliers if there is a possible way around that obvious curve
# TODO

amas_nobl.to_csv('AMAS_nobl_cleaned.csv')


# Also from another script we see that the first year is quite unstable. So I am removing the first 2 visits' data
# The problem is that now we only have 3341 videos (scans)
amas_nobl = amas_nobl[amas_nobl['years'] >= 1]
amas_nobl.to_csv('AMAS_nobl_nofirstyear.csv')