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


# Print the first few rows of the merged dataset
print(merged_df.head())
#info
merged_df.info()
#missing value
missing_values = merged_df.isnull().sum()

# Display columns with missing values
print(missing_values)

#use log transformation for screentime data
screentime = merged_df[['C_wk', 'G_wk', 'S_wk', 'T_wk', 'C_we', 'G_we', 'S_we', 'T_we']]
for column in screentime:
    merged_df[f'{column}_log'] = np.log(merged_df[column] + 1)

#Feature Engineering
merged_df['Total_Wellbeing'] = (merged_df['Optm'] +
    merged_df['Usef'] +
    merged_df['Relx'] +
    merged_df['Intp'] +
    merged_df['Engs'] +
    merged_df['Dealpr']+
    merged_df['Thcklr'] +
    merged_df['Goodme'] +
    merged_df['Clsep'] +
    merged_df['Conf'] +
    merged_df['Mkmind'] +
    merged_df['Loved']+
    merged_df['Intthg'] +
    merged_df['Cheer'])

# Compute Average Well-Being Score
merged_df['Average_Well_Being_Score'] = merged_df['Total_Wellbeing']/14

print(merged_df['Average_Well_Being_Score'].head())

# Save the merged dataset
merged_df.to_csv('merged_dataset.csv', index=False)