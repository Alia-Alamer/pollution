import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

with open('feinstaubdataexercise.pickle', 'rb') as file:
    content = pkl.load(file)

df_graz = content['Graz-DB']
df_kalk = content['Kalkleiten']

df_graz.index = pd.to_datetime(df_graz.index)
df_kalk.index = pd.to_datetime(df_kalk.index)

df = df_graz.merge(df_kalk[['temp']], left_index=True, right_index=True, 
                  suffixes=('_graz', '_kalk'))

df['temp_diff'] = df['temp_graz'] - df['temp_kalk']
df['frost'] = (df['temp_graz'] < 0).astype(int)
df['inversion'] = (df['temp_diff'] < 0).astype(int)
df['strong_wind'] = (df['windspeed'] > 0.6).astype(int)

df['temp_graz_lag1'] = df['temp_graz'].shift(1)
df['humidity_lag1'] = df['humidity'].shift(1)
df['windspeed_lag1'] = df['windspeed'].shift(1)

df = df.dropna()

train = df[df.index.year <= 2019].copy()
test = df[df.index.year == 2020].copy()

train['sqrt_pm10'] = np.sqrt(train['pm10'])
train['sqrt_no2'] = np.sqrt(train['no2'])

formula = " ~ C(day_type) + temp_graz + prec + windspeed + peak_velocity + temp_diff + frost + temp_graz_lag1 + humidity_lag1 + windspeed_lag1"

model_pm10 = smf.ols("sqrt_pm10" + formula, data=train).fit()
model_no2 = smf.ols("sqrt_no2" + formula, data=train).fit()

print(f"PM10 R²: {model_pm10.rsquared:.3f}")
print(f"NO2 R²: {model_no2.rsquared:.3f}")

test['pred_pm10'] = model_pm10.predict(test) ** 2
test['pred_no2'] = model_no2.predict(test) ** 2

print("\nFirst 5 predictions:")
print(test[['pred_pm10', 'pred_no2']].head())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(test.index, test['pred_pm10'], label='Predicted PM10', color='red', linewidth=1)
ax1.set_ylabel('PM10 (µg/m³)')
ax1.set_title('PM10 Predictions for 2020')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(test.index, test['pred_no2'], label='Predicted NO2', color='orange', linewidth=1)
ax2.set_ylabel('NO2 (µg/m³)')
ax2.set_title('NO2 Predictions for 2020')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.xlabel('Date')
plt.tight_layout()
plt.show()

test[['pred_pm10', 'pred_no2']].to_csv('predictions_2020.csv')
print("\nPredictions saved to 'predictions_2020.csv'")