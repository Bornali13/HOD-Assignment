import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

chunk_size = 15000  # Adjust the chunk size as needed
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

###Univariate Analysis###

#Screentime
screentime_log = merged_df[['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log']]
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
sns.distplot(merged_df['avg_wellbeing_log'])
plt.subplot(1,2,2)
sns.boxplot(merged_df['avg_wellbeing_log'])
plt.close()


from scipy.stats import spearmanr

# Calculate pearson correlation 
# Select the columns to compute correlations (e.g., log-transformed screentime and well-being)
columns_for_correlation = ['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log', 'avg_wellbeing_log']

# Calculate the Pearson correlation matrix
correlation_matrix = merged_df[columns_for_correlation].corr(method='pearson')

# Display the correlation matrix
print(correlation_matrix)

# Optionally, visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Pearson Correlation Matrix')
plt.show()


