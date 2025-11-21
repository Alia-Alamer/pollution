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
