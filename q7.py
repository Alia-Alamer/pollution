import pandas as pd
import matplotlib.pyplot as plt

predictions = pd.read_csv('predictions_2020.csv', index_col=0, parse_dates=True)

with open('feinstaubdataexercise.pickle', 'rb') as file:
    content = pd.read_pickle(file)

df_graz = content['Graz-DB']
df_graz.index = pd.to_datetime(df_graz.index)

actual_2020 = df_graz[df_graz.index.year == 2020][['pm10', 'no2']]

results = actual_2020.copy()
results['pm10_pred'] = predictions['pred_pm10']
results['no2_pred'] = predictions['pred_no2']

first_six = results.loc['2020-01-01':'2020-06-30']

lockdowns = [
    {'start': '2020-03-16', 'end': '2020-04-14', 'label': '1. Hard Lockdown', 'color': 'red'},
    {'start': '2020-04-15', 'end': '2020-05-01', 'label': 'Lockerungen Phase 1', 'color': 'orange'},
    {'start': '2020-05-02', 'end': '2020-05-15', 'label': 'Lockerungen Phase 2', 'color': 'yellow'},
    {'start': '2020-05-16', 'end': '2020-06-30', 'label': 'Lockerungen Phase 3', 'color': 'lime'}
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(first_six.index, first_six['pm10'], label='Actual PM10', color='blue', linewidth=1)
ax1.plot(first_six.index, first_six['pm10_pred'], label='Predicted PM10', color='red', linestyle='--', linewidth=1)

ax2.plot(first_six.index, first_six['no2'], label='Actual NO2', color='green', linewidth=1)
ax2.plot(first_six.index, first_six['no2_pred'], label='Predicted NO2', color='orange', linestyle='--', linewidth=1)

for ax in [ax1, ax2]:
    for lockdown in lockdowns:
        start = pd.to_datetime(lockdown['start'])
        end = pd.to_datetime(lockdown['end'])
        ax.axvspan(start, end, alpha=0.2, color=lockdown['color'], label=lockdown['label'])

ax1.set_ylabel('PM10 (µg/m³)')
ax1.set_title('PM10: Actual vs Predicted (Jan-Jun 2020)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_ylabel('NO2 (µg/m³)')
ax2.set_title('NO2: Actual vs Predicted (Jan-Jun 2020)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.xlabel('Date')
plt.tight_layout()
plt.show()