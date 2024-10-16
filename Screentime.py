import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

screentime = merged_df[['C_wk', 'G_wk', 'S_wk', 'T_wk', 'C_we', 'G_we', 'S_we', 'T_we']]
screentime.describe()

#Frequency of Distribution
screentime.mean()

#Nature of Distribution
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

screentime_log_transformed = np.log1p(screentime)
screentime_log_transformed.describe()
print(screentime_log_transformed)

plot_warning(screentime_log_transformed)

#Shapiro-Wilk test
from scipy import stats

# Test for normality (Shapiro-Wilk test)
stat, p_value = stats.shapiro(screentime_log_transformed['C_wk'])
print(f'Statistic: {stat}, p-value: {p_value}')