import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

with open('feinstaubdataexercise.pickle', 'rb') as file:
    content = pkl.load(file)

df_graz = content['Graz-DB']
df_kalk = content['Kalkleiten']

df_graz.index = pd.to_datetime(df_graz.index)
df_kalk.index = pd.to_datetime(df_kalk.index)

df_graz = df_graz[(df_graz.index.year >= 2015) & (df_graz.index.year <= 2019)]
df_kalk = df_kalk[(df_kalk.index.year >= 2015) & (df_kalk.index.year <= 2019)]

plt.figure(figsize=(10,4))
plt.plot(df_graz.index, df_graz['temp'], label='Graz')
plt.plot(df_kalk.index, df_kalk['temp'], label='Kalkleiten')
plt.xlabel('Datum')
plt.ylabel('Temperatur (°C)')
plt.legend()
plt.title('Temperaturen Graz vs. Kalkleiten')
plt.show()

df = df_graz.merge(df_kalk[['temp']], left_index=True, right_index=True, suffixes=('_graz', '_kalk'))
df['temp_diff'] = df['temp_graz'] - df['temp_kalk']
print("Temperaturdifferenz Graz-Kalkleiten:")

print(df['temp_diff'].describe())

model_pm10 = smf.ols(
    "pm10 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity",
    data=df
).fit()

model_no2 = smf.ols(
    "no2 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity",
    data=df
).fit()

print("\n=== PM10 Modell ===")
print(model_pm10.summary())

print("\n=== NO2 Modell ===")
print(model_no2.summary())

model_pm10_diff = smf.ols(
    "pm10 ~ C(day_type) + humidity + prec + windspeed + peak_velocity + temp_diff",
    data=df
).fit()

model_no2_diff = smf.ols(
    "no2 ~ C(day_type) + humidity + prec + windspeed + peak_velocity + temp_diff",
    data=df
).fit()

print("\n=== PM10 Modell mit temp_diff ===")
print(model_pm10_diff.summary())

print("\n=== NO2 Modell mit temp_diff ===")
print(model_no2_diff.summary())
