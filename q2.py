import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl


with open ('feinstaubdataexercise.pickle', 'rb') as file:
    filecontent = pkl.load(file)


content = filecontent["Graz-DB"]

model_pm10 = smf.ols(formula="pm10 ~ C(day_type) + humidity + temp + prec + windspeed + peak_velocity", data=content).fit()
print(model_pm10.summary())

print(anova_lm(model_pm10))



model_no2 = smf.ols(formula="no2 ~ C(day_type) + humidity + temp + prec + windspeed + peak_velocity", data=content).fit()
print(model_no2.summary())

print(anova_lm(model_no2))




sns.scatterplot(x=model_pm10.fittedvalues, y=model_pm10.resid)
plt.axhline(0, color="red")
plt.xlabel('fitted values (vorhergesagte werte)')
plt.ylabel('residuals (fehler)')
plt.title("pm10 residuen vs. fitted")
plt.show()


sns.scatterplot(x=model_no2.fittedvalues, y=model_no2.resid)
plt.axhline(0, color="red")
plt.xlabel('fitted values (vorhergesagte werte)')
plt.ylabel('residuals (fehler)')
plt.title("no2 residuen vs. fitted")
plt.show()


sm.qqplot(model_pm10.resid, line='s')
plt.title("PM10 QQ-Plot")
plt.show()

sm.qqplot(model_no2.resid, line='s')
plt.title("NO2 QQ-Plot")
plt.show()