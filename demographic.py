import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000
chunks = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/merged_dataset.csv', encoding='utf-8', chunksize=chunk_size)
merged_df = pd.concat(chunks)

#Frequency Distribution
demographic = merged_df[['gender', 'minority', 'deprived']]
demographic.mode()

#Demographic data maping
#gender
merged_df['gender_mapped'] = merged_df['gender'].map({1: 'Male', 0: 'Female'})
#Minority
merged_df['minority_mapped'] = merged_df['minority'].map({1: 'Minority', 0: 'Mejority'})
#Deprived
merged_df['deprived_mapped'] = merged_df['deprived'].map({1: 'In locality', 0: 'Outside locality'})

mapped_demographic = merged_df[['gender_mapped', 'minority_mapped', 'deprived_mapped']]
mapped_demographic.head()


# Descriptive analysis and visualization
def plot_demographic_distribution(df, column, title):
    # Frequency counts
    counts = df[column].value_counts()

    # Bar Plot
    plt.figure(figsize=(8, 5))
    counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title(f'{title} - Bar Plot')
    plt.ylabel('Count')
    plt.xlabel(column)
    plt.xticks(rotation=0)
    plt.show()

    # Pie Chart
    plt.figure(figsize=(6, 6))
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightblue'], startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title(f'{title} - Pie Chart')
    plt.ylabel('')  # Remove y-label for pie chart
    plt.show()

# Gender distribution
plot_demographic_distribution(merged_df, 'gender_mapped', 'Gender Distribution')

# Minority status distribution
plot_demographic_distribution(merged_df, 'minority_mapped', 'Minority Status Distribution')

# Deprivation status distribution
plot_demographic_distribution(merged_df, 'deprived_mapped', 'Deprivation Status Distribution')
