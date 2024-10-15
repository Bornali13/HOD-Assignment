import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

chunk_size = 15000  # Adjust the chunk size as needed
df1 = pd.read_csv('C:\Users\Asus\OneDrive\Documents\GitHub\HOD-Assignment\dataset1.csv', encoding='utf-8', chunksize=chunk_size)
df2 = pd.read_csv('C:\Users\Asus\OneDrive\Documents\GitHub\HOD-Assignment\dataset2.csv', encoding='utf-8', chunksize=chunk_size)
df3 = pd.read_csv('C:\Users\Asus\OneDrive\Documents\GitHub\HOD-Assignment\dataset3.csv', encoding='utf-8', chunksize=chunk_size)

