import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000  # Adjust the chunk size as needed
chunks3 = pd.read_csv('C:/Users/Asus/OneDrive/Documents/GitHub/HOD-Assignment/dataset3.csv', encoding='utf-8', chunksize=chunk_size)
wellbeing = pd.concat(chunks3)

# Calculate  mode for each column
statistics = {}
for col in wellbeing.columns[1:]:  # Exclude 'ID' column
    statistics[col] = {
        'mode': wellbeing[col].mode()[0]  # Mode might return multiple values, so take the first
    }

# Convert to DataFrame for better visualization
stats_df = pd.DataFrame(statistics)

print(stats_df)

# Drop the 'ID' column
wellbeing = wellbeing.drop('ID', axis=1)

mapped_wellbeing = wellbeing.replace({1: 'None of the time', 2: 'Rarely' , 3: 'Some of the times',
                                      4: 'Often', 5: 'All of the time'})

def plot_wellbeing_distribution(df, column, title):
    # Frequency counts
    counts = df[column].value_counts()

    # Calculate the percentage distribution
    percentage_distribution = (counts / len(df[column])) * 100 # Changed 'frequency_distribution' to 'counts'

    # Plot the bar chart
    percentage_distribution.plot(kind='bar', color='skyblue')

    # Add labels and title
    plt.xlabel('Categories')
    plt.ylabel('Percentage (%)')
    plt.title(title)

    # Display percentages on top of bars
    for i in range(len(percentage_distribution)):
        plt.text(i, percentage_distribution[i] + 0.5, f'{percentage_distribution[i]:.2f}%', ha='center', va='bottom')

    # Save the plot as an image file
    plt.savefig(f'{column}_wellbeing_distribution.png', bbox_inches='tight')

    # Close the plot to avoid overlapping of multiple figures
    plt.close()

# Call the function to plot the bar chart with percentages
plot_wellbeing_distribution(mapped_wellbeing, 'Optm', 'Wellbeing Distribution for Optm')
plot_wellbeing_distribution(mapped_wellbeing, 'Usef', 'Wellbeing Distribution for Usef')
plot_wellbeing_distribution(mapped_wellbeing, 'Relx', 'Wellbeing Distribution for Relx')
plot_wellbeing_distribution(mapped_wellbeing, 'Intp', 'Wellbeing Distribution for Intp')
plot_wellbeing_distribution(mapped_wellbeing, 'Engs', 'Wellbeing Distribution for Engs')
plot_wellbeing_distribution(mapped_wellbeing, 'Dealpr', 'Wellbeing Distribution for Dealpr')
plot_wellbeing_distribution(mapped_wellbeing, 'Thcklr', 'Wellbeing Distribution for Thcklr')
plot_wellbeing_distribution(mapped_wellbeing, 'Goodme', 'Wellbeing Distribution for Goodme')
plot_wellbeing_distribution(mapped_wellbeing, 'Clsep', 'Wellbeing Distribution for Clsep')
plot_wellbeing_distribution(mapped_wellbeing, 'Conf', 'Wellbeing Distribution for Conf')
plot_wellbeing_distribution(mapped_wellbeing, 'Mkmind', 'Wellbeing Distribution for Mkmind')
plot_wellbeing_distribution(mapped_wellbeing, 'Loved', 'Wellbeing Distribution for Loved')
plot_wellbeing_distribution(mapped_wellbeing, 'Intthg', 'Wellbeing Distribution for Intthg')
plot_wellbeing_distribution(mapped_wellbeing, 'Cheer', 'Wellbeing Distribution for Cheer')

