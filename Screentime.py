import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000
chunks2 = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/dataset2.csv', encoding='utf-8', chunksize=chunk_size)
screentime = pd.concat(chunks2)
screentime.describe()


# Calculating mean, median, and mode for each column
mean_values = screentime.mean()
median_values = screentime.median()
mode_values = screentime.mode().iloc[0]

# Prepare the results
summary_stats = pd.DataFrame({
    'Mean': mean_values,
    'Median': median_values,
    'Mode': mode_values
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Screen Time Summary Statistics", dataframe=summary_stats)

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