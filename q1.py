import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open ('feinstaubdataexercise.pickle', 'rb') as file:
    filecontent = pkl.load(file)

#print(filecontent)
data = filecontent["Graz-DB"]
#print(data.head())

print("dataset shape:", data.shape)
print("\n spalten:", data.columns.tolist())
print(f"von {data.index.min()} bis {data.index.max()}")

print("=========================")

print("\n statistiken der response variablen:")
print(data[['pm10', 'no2']].describe())

fig, axes = plt.subplots(2, 1, figsize = (15, 10))

axes[0].plot(data.index, data['pm10'], alpha=0.7)
axes[0].set_title('pm10 Konzentration über Zeit')
axes[0].set_ylabel('pm10 (μg/m³)')
#axes[0].set_xlabel('Datum')
axes[0].grid(True)

axes[1].plot(data.index, data['no2'], alpha=0.7, color='pink')
axes[1].set_title('no2 Konzentration über Zeit')
axes[1].set_ylabel('no2 (μg/m³)')
axes[1].set_xlabel('Datum')
axes[1].grid(True)

plt.tight_layout()
plt.show()