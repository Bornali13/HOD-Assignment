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

##Screentime Data##

###Handling Outliers###
#Log transform
# List of screentime columns
screentime = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']

merged_df[screentime] = merged_df[screentime].replace(0, np.nan)  # Replace zeros with NaN to handle missing data
merged_df[screentime] = merged_df[screentime].fillna(1)  # Fill NaN with 1 before log transformation

# Apply log transformation to each screentime column and store in a new column
for column in screentime:
    merged_df[f'{column}_log'] = np.log(merged_df[column] + 1)  # Adding 1 to avoid log(0)

# Function to remove outliers using the IQR method
screentime_log = merged_df[['C_we_log', 'C_wk_log', 'G_we_log', 'G_wk_log', 'S_we_log', 'S_wk_log', 'T_we_log', 'T_wk_log']]

# Function to remove outliers using the IQR method
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

# Apply the IQR outlier removal to the screentime log-transformed columns
screentime_cleaned_data = remove_outliers_iqr(screentime_log.copy(), screentime_log.columns)

# Step 3: Add the cleaned screentime data back to the original merged_df
# Make sure to update only the relevant columns
merged_df.update(screentime_cleaned_data)


###Feature Engineering###
##Well-being##
# Well-being columns (replace these with your actual column names)
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Calculate the average well-being score
merged_df['avg_wellbeing'] = merged_df[wellbeing_columns].mean(axis=1)

# Apply log transformation to the 'avg_wellbeing' column
# Add 1 to avoid log(0) in case there are zero values
merged_df['avg_wellbeing_log'] = np.log(merged_df['avg_wellbeing'] + 1)

# Display the first few rows with the new log-transformed column
print(merged_df[['avg_wellbeing', 'avg_wellbeing_log']].head())

# Apply to avg_wellbeing columns
# Assuming merged_df already has the 'avg_wellbeing_log' column

# Step 1: Calculate the 1st quartile (Q1) and 3rd quartile (Q3)
Q1 = merged_df['avg_wellbeing_log'].quantile(0.25)
Q3 = merged_df['avg_wellbeing_log'].quantile(0.75)

# Step 2: Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Step 3: Define the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4: Filter the DataFrame to remove outliers
filtered_df = merged_df[(merged_df['avg_wellbeing_log'] >= lower_bound) & (merged_df['avg_wellbeing_log'] <= upper_bound)]

# align indices (if necessary)
filtered_df = filtered_df.reindex(merged_df.index)

# Then update the column
merged_df['avg_wellbeing_log'] = filtered_df['avg_wellbeing_log']

#### Save the merged dataset####
merged_df.to_csv('merged_dataset.csv', index=False)
