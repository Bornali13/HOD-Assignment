import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000  # Adjust the chunk size as needed
chunks1 = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/dataset1.csv', encoding='utf-8', chunksize=chunk_size)
chunks2 = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/dataset2.csv', encoding='utf-8', chunksize=chunk_size)
chunks3 = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/dataset3.csv', encoding='utf-8', chunksize=chunk_size)

# Combine the chunks into a single dataframe
df1 = pd.concat(chunks1)
df2 = pd.concat(chunks2)
df3 = pd.concat(chunks3)

# Merge the datasets
merged_df = pd.merge(df1, df2, on='ID', how='inner')
merged_df = pd.merge(merged_df, df3, on='ID', how='inner')

# Save the merged dataset
merged_df.to_csv('merged_dataset.csv', index=False)

# Print the first few rows of the merged dataset
print(merged_df.head())


