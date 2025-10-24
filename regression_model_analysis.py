# Завдання: побудувати регресійну модель (лінійну множинну) щоб встановити вплив змінних (glucose, LDL, HDL, VLDL, views)
# на гемоглобін; оцінити модель (коефіцієнти, p-value, R2, VIF) та побудувати рівняння регресії

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Генерація синтетики (поєднання попередніх даних)
np.random.seed(2)
n = 200
glucose = np.random.uniform(70, 180, n)
ldl = np.random.normal(130, 28, n)
hdl = np.random.normal(50, 10, n)
vldl = np.random.normal(30, 7, n)
views = np.random.poisson(18, n)  # поведінковий показник

# Залежна змінна Hemoglobin — комбінація факторів
hb = 12.8 + 0.004*(ldl-120) - 0.002*(hdl-50) - 0.001*(vldl-30) - 0.003*(glucose-100) + 0.02*(views-18) + np.random.normal(0, 0.3, n)

df = pd.DataFrame({
    'Hemoglobin': hb,
    'glucose': glucose,
    'LDL': ldl,
    'HDL': hdl,
    'VLDL': vldl,
    'views': views
})

# 1) Побудова лінійної моделі (OLS)
X = df[['glucose','LDL','HDL','VLDL','views']]
X = sm.add_constant(X)  # додаємо константу
y = df['Hemoglobin']

model = sm.OLS(y, X).fit()
print(model.summary())

# 2) Обчислення VIF для виявлення мультиколінеарності
vif_df = pd.DataFrame()
vif_df['feature'] = X.columns
vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n=== VIF ===")
print(vif_df)

# 3) Рівняння регресії (математичне рівняння)
coefs = model.params
equation = f"Hemoglobin = {coefs['const']:.4f}"
for var in ['glucose','LDL','HDL','VLDL','views']:
    equation += f" + ({coefs[var]:+.4f})*{var}"
print("\n=== Регресійне рівняння ===")
print(equation)

# 4) Оцінка впливу змінних (t-values, p-values)
print("\n=== Вплив змінних (t, p) ===")
print(model.tvalues)
print(model.pvalues)

# 5) Діагностика залишків
resid = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.scatterplot(x=fitted, y=resid, s=20)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

plt.subplot(1,2,2)
sns.histplot(resid, kde=True)
plt.title('Residuals distribution')
plt.tight_layout()
plt.savefig("regression_diagnostics.png")
print("\nДіагностичні графіки збережено як 'regression_diagnostics.png'")

# 6) Короткий висновок
r2 = model.rsquared
adj_r2 = model.rsquared_adj
print(f"\nR^2 = {r2:.4f}, Adjusted R^2 = {adj_r2:.4f}")