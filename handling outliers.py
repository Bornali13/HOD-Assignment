
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000  # Adjust the chunk size as needed
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

#Log transform
# List of screentime columns
screentime = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']

# Apply log transformation to each screentime column and store in a new column
for column in screentime:
    merged_df[f'{column}_log'] = np.log(merged_df[column] + 1)  # Adding 1 to avoid log(0)

# Display the first few rows of the dataset with the new log-transformed columns
print(merged_df.head())

# Function to remove outliers using the IQR method

screentime_log = merged_df[['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log']]
def remove_outliers_iqr(df, columns):
    for column in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the data to remove outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

# Apply to screentime columns
cleaned_data = remove_outliers_iqr(merged_df, screentime_log)

# Display cleaned data
print(cleaned_data)

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
plot_warning(screentime_log)