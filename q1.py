import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open ('feinstaubdataexercise.pickle', 'rb') as file:
    filecontent = pkl.load(file)

#print(filecontent)
content = filecontent["Graz-DB"]
#print(data.head())

print("dataset shape:", content.shape)
print("\n spalten:", content.columns.tolist())
print(f"von {content.index.min()} bis {content.index.max()}")

print("=========================")

print("\n statistiken der response variablen:")
print(content[['pm10', 'no2']].describe())

fig, axes = plt.subplots(2, 1, figsize = (15, 10))

axes[0].plot(content.index, content['pm10'], alpha=0.7)
axes[0].set_title('pm10 Konzentration über Zeit')
axes[0].set_ylabel('pm10 (μg/m³)')
#axes[0].set_xlabel('Datum')
axes[0].grid(True)

axes[1].plot(content.index, content['no2'], alpha=0.7, color='pink')
axes[1].set_title('no2 Konzentration über Zeit')
axes[1].set_ylabel('no2 (μg/m³)')
axes[1].set_xlabel('Datum')
axes[1].grid(True)

plt.tight_layout()
plt.show()


predictors = ['temp', 'humidity', 'windspeed', 'prec', 'peak_velocity']

for preds in predictors:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=content, x=preds, y="pm10")
    plt.title(f"pm10 vs {preds}")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.scatterplot(data=content, x=preds, y="no2")
    plt.title(f"no2 vs {preds}")
    plt.show()

