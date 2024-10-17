import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000  # Adjust the chunk size as needed
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

##Univariate Analysis:

#Screentime
 
screentime_log = merged_df[['C_wk_log', 'G_wk_log', 'S_wk_log', 'T_wk_log', 'C_we_log', 'G_we_log', 'S_we_log', 'T_we_log']]
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
          plt.close()
plot_warning(screentime_log)

#wellbeing

import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(merged_df['Average_Well_Being_Score'])
plt.subplot(1,2,2)
sns.boxplot(merged_df['Average_Well_Being_Score'])
plt.show()


Average_Well_Being_Score_log_transformed = np.log1p(merged_df['Average_Well_Being_Score'])
Average_Well_Being_Score_log_transformed.describe()
print(Average_Well_Being_Score_log_transformed)
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(Average_Well_Being_Score_log_transformed)
plt.subplot(1,2,2)
sns.boxplot(Average_Well_Being_Score_log_transformed)
plt.show()