# Завдання: підібрати параметри нелінійної моделі для аналізу зв'язку між "stabilized_glucose" (незалежна) та "Hemoglobin" (залежна),
# порівняти кілька форм (експоненційна, поліноміальна) і оцінити якість (R2, RMSE).

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1) Синтетичні дані (stabilized_glucose vs Hemoglobin)
np.random.seed(1)
n = 120
glucose = np.random.uniform(70, 180, n)  # мг/дл
# Приклад нелінійної залежності: Hb = a + b * exp(-c * glucose) + noise
true_a, true_b, true_c = 13.0, -2.5, 0.01
hb = true_a + true_b * np.exp(-true_c * glucose) + np.random.normal(0, 0.2, n)

df = pd.DataFrame({'glucose': glucose, 'Hemoglobin': hb})

# 2) Визначимо функції моделей
def exp_model(x, a, b, c):
    return a + b * np.exp(-c * x)

def poly2(x, a, b, c):
    return a + b*x + c*(x**2)

def michaelis_menten(x, Vmax, Km, baseline):
    # приклад насичувальної функції
    return baseline + (Vmax * x) / (Km + x)

models = {
    'exponential': (exp_model, [12.5, -2.0, 0.01]),
    'polynomial2': (poly2, [10.0, 0.01, 0.0001]),
    'michaelis_menten': (michaelis_menten, [2.0, 100.0, 12.0])
}

results = {}

# 3) Підбір параметрів методом найменших квадратів (curve_fit)
for name, (func, p0) in models.items():
    try:
        popt, pcov = curve_fit(func, df['glucose'], df['Hemoglobin'], p0=p0, maxfev=10000)
        residuals = df['Hemoglobin'] - func(df['glucose'], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((df['Hemoglobin'] - np.mean(df['Hemoglobin']))**2)
        r2 = 1 - ss_res/ss_tot
        rmse = np.sqrt(mean_squared_error(df['Hemoglobin'], func(df['glucose'], *popt)))
        perr = np.sqrt(np.diag(pcov))
        results[name] = {'popt': popt, 'perr': perr, 'r2': r2, 'rmse': rmse}
    except Exception as e:
        results[name] = {'error': str(e)}

# 4) Вивід результатів
for name, res in results.items():
    print(f"\n=== Model: {name} ===")
    if 'error' in res:
        print("Error:", res['error'])
        continue
    print("Params:", np.round(res['popt'],5))
    print("Std errors:", np.round(res['perr'],5))
    print("R^2:", round(res['r2'],4))
    print("RMSE:", round(res['rmse'],4))

# 5) Візуалізація найкращої моделі (вибираємо по R^2)
best = max(( (k, v['r2']) for k,v in results.items() if 'r2' in v), key=lambda x: x[1])[0]
best_func = models[best][0]
best_popt = results[best]['popt']

plt.figure(figsize=(8,5))
plt.scatter(df['glucose'], df['Hemoglobin'], s=20, label='data')
x_line = np.linspace(df['glucose'].min(), df['glucose'].max(), 200)
plt.plot(x_line, best_func(x_line, *best_popt), color='red', label=f'Best fit: {best}')
plt.xlabel('Stabilized glucose (mg/dL)')
plt.ylabel('Hemoglobin')
plt.title(f'Nonlinear fit ({best})')
plt.legend()
plt.tight_layout()
plt.savefig("nonlinear_glucose_hb_fit.png")
print(f"\nBest model: {best}. Plot saved as 'nonlinear_glucose_hb_fit.png'")