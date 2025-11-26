import pickle as pkl
import pandas as pd
import statsmodels.formula.api as smf


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

#PO10
model_pm10 = smf.ols(
    "pm10 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity",
    data=df
).fit()

model_pm10_frost = smf.ols(
    "pm10 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity + \
     temp_diff + frost",
    data=df
).fit()

# #Diese Variable verschlechtert das Model um 1.
# # model_pm10_inversion = smf.ols(
# #     "pm10 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity + \
# #      temp_diff + frost + inversion",
# #     data=df
# # ).fit()

# # #Diese Variable verschlechtert das Model um 1.
# # model_pm10_strong_wind = smf.ols(
# #     "pm10 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity + \
# #      temp_diff + frost + strong_wind",
# #     data=df
# # ).fit()

# # #Diese Variable macht keinen Unterschied beim Model.
# # model_pm10_year = smf.ols(
# #     "pm10 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity + \
# #      temp_diff + frost + year",
# #     data=df
# # ).fit()

model_pm10_temp_graz_lag1 = smf.ols(
    "pm10 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity + \
     temp_diff + frost + temp_graz_lag1",
    data=df
).fit()


model_pm10_humidity_lag1 = smf.ols(
    "pm10 ~ C(day_type) + temp_graz + prec + windspeed + peak_velocity + \
     temp_diff + frost + temp_graz_lag1 + humidity_lag1",
    data=df
).fit()

model_pm10_windspeed_lag1 = smf.ols(
    "pm10 ~ C(day_type) + humidity + temp_graz + prec + \
     temp_diff+ windspeed_lag1",
    data=df
).fit()

model_pm10_all = smf.ols(
    "pm10 ~ C(day_type) + temp_graz + prec + windspeed + peak_velocity + \
      temp_diff + frost + temp_graz_lag1 + humidity_lag1 + windspeed_lag1",
    data=df
).fit()


# #NO2 Modell

model_no2 = smf.ols(
    "no2 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity",
    data=df
).fit()

model_no2_inversion = smf.ols(
     "no2 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity + \
      temp_diff + frost + inversion",
     data=df
 ).fit()

#Hilft dem Modell nicht
# model_no2_strong_wind = smf.ols(
#     "no2 ~ C(day_type) + humidity + temp_graz + prec + windspeed + peak_velocity + \
#      temp_diff + inversion + strong_wind",
#     data=df
# ).fit()

model_no2_humidity_lag1 = smf.ols(
     "no2 ~ C(day_type) + temp_graz + prec + windspeed + peak_velocity + \
     temp_diff + humidity_lag1",
     data=df
 ).fit() 


model_no2_windspeed_lag1 = smf.ols(
     "no2 ~ C(day_type) + humidity + temp_graz + prec + peak_velocity + \
      temp_diff +windspeed_lag1",
     data=df
 ).fit()

model_no2_all = smf.ols(
    "no2 ~ C(day_type) + temp_graz + prec + windspeed + peak_velocity + \
     temp_diff + inversion + humidity_lag1 + windspeed_lag1",
    data=df
).fit()

# PM10
print(1, model_pm10.summary())
print(2, model_pm10_frost.summary())
print(3, model_pm10_temp_graz_lag1.summary())
print(4, model_pm10_humidity_lag1.summary())
print(5, model_pm10_windspeed_lag1.summary())
print(1, model_pm10_all.summary())


# NO2
print(6, model_no2.summary())
print(8, model_no2_inversion.summary())
print(11, model_no2_humidity_lag1.summary())
print(12, model_no2_windspeed_lag1.summary())
print(6, model_no2_all.summary())

