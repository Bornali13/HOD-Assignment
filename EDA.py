import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000  # Adjust the chunk size as needed
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

##Univariate Analysis:

#Screentime
 
screentime = merged_df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]
import warnings
warnings.filterwarnings('ignore')

def plot_warning(df):
    # Loop through each column in the DataFrame
    for column in df.columns:
        # Check if the column contains numeric data
        if pd.api.types.is_numeric_dtype(df[column]):
          plt.figure(figsize=(16,5))
          plt.subplot(1,2,1)
          sns.distplot(df[column],bins=7)
          plt.subplot(1,2,2)
          sns.boxplot(df[column])
          plt.show()
plot_warning(screentime)

#wellbeing

import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(merged_df['Average_Well_Being_Score'])
plt.subplot(1,2,2)
sns.boxplot(merged_df['Average_Well_Being_Score'])
plt.show()

