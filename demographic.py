import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000
chunks1 = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/dataset1.csv', encoding='utf-8', chunksize=chunk_size)
demographic = pd.concat(chunks1)

# Calculate  mode for each column
statistics = {}
for col in demographic.columns[1:]:  # Exclude 'ID' column
    statistics[col] = {
        'mode': demographic[col].mode()[0]  # Mode might return multiple values, so take the first
    }

# Convert to DataFrame for better visualization
stats_df = pd.DataFrame(statistics)

print(stats_df)

#Demographic data maping
#gender
demographic['gender_mapped'] = demographic['gender'].map({1: 'Male', 0: 'Female'})
#Minority
demographic['minority_mapped'] = demographic['minority'].map({1: 'Minority', 0: 'Mejority'})
#Deprived
demographic['deprived_mapped'] = demographic['deprived'].map({1: 'In locality', 0: 'Outside locality'})

mapped_demographic = demographic[['gender_mapped', 'minority_mapped', 'deprived_mapped']]
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
plot_demographic_distribution(demographic, 'gender_mapped', 'Gender Distribution')

# Minority status distribution
plot_demographic_distribution(demographic, 'minority_mapped', 'Minority Status Distribution')

# Deprivation status distribution
plot_demographic_distribution(demographic, 'deprived_mapped', 'Deprivation Status Distribution')
