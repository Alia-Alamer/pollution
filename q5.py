import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle as pkl
import pandas as pd


with open('feinstaubdataexercise.pickle', 'rb') as file:
    content = pkl.load(file)

df_graz = content['Graz-DB']
df_kalk = content['Kalkleiten']

df_graz.index = pd.to_datetime(df_graz.index)
df_kalk.index = pd.to_datetime(df_kalk.index)

df_graz = df_graz[(df_graz.index.year >= 2015) & (df_graz.index.year <= 2019)]
df_kalk = df_kalk[(df_kalk.index.year >= 2015) & (df_kalk.index.year <= 2019)]

df = df_graz.merge(df_kalk[['temp']], left_index=True, right_index=True, suffixes=('_graz', '_kalk'))
df['temp_diff'] = df['temp_graz'] - df['temp_kalk']

df['frost'] = (df['temp_graz'] < 0).astype(int)
df['inversion'] = (df['temp_diff'] < 0).astype(int)
df['strong_wind'] = (df['windspeed'] > 0.6).astype(int)
df['year'] = df.index.year

df['temp_graz_lag1'] = df['temp_graz'].shift(1)
df['humidity_lag1'] = df['humidity'].shift(1)
df['windspeed_lag1'] = df['windspeed'].shift(1)

df = df.dropna()

df['sqrt_pm10'] = np.sqrt(df['pm10'])

model_pm10_sqrt = smf.ols(
    "sqrt_pm10 ~ C(day_type) + temp_graz + prec + windspeed + peak_velocity + temp_diff + frost + temp_graz_lag1 + humidity_lag1 + windspeed_lag1",
    data=df
).fit()

print(model_pm10_sqrt.summary())

residuals = model_pm10_sqrt.resid
fitted = model_pm10_sqrt.fittedvalues

plt.figure(figsize=(8,5))
sns.scatterplot(x=fitted, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (sqrt PM10)')
plt.show()

sm.qqplot(residuals, line='45')
plt.title('QQ-Plot der Residuen (sqrt PM10)')
plt.show()



df['sqrt_no2'] = np.sqrt(df['no2'])

model_no2_sqrt = smf.ols(
    "sqrt_no2 ~ C(day_type) + temp_graz + prec + windspeed + peak_velocity + temp_diff + frost + temp_graz_lag1 + humidity_lag1 + windspeed_lag1",
    data=df
).fit()

print(model_no2_sqrt.summary())

residuals = model_no2_sqrt.resid
fitted = model_no2_sqrt.fittedvalues

plt.figure(figsize=(8,5))
sns.scatterplot(x=fitted, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (sqrt no2)')
plt.show()

sm.qqplot(residuals, line='45')
plt.title('QQ-Plot der Residuen (sqrt no2)')
plt.show()

