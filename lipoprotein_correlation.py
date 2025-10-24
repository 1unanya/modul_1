# Завдання: обчислення коефіцієнтів кореляції між змінними ліпопротеїнів (LDL, HDL, VLDL, TotalChol) та гемоглобіном (Hb)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Генеруємо приклад медичних даних (синтетика)
np.random.seed(0)
n = 150
# Основні змінні ліпопротеїнів
ldl = np.random.normal(130, 30, n)  # LDL
hdl = np.random.normal(50, 12, n)   # HDL
vldl = np.random.normal(30, 8, n)   # VLDL
total_chol = 0.8*ldl + 0.2*hdl + np.random.normal(0, 10, n)

# Гемоглобін залежить слабко від ліпопротеїнів + шум
hb = 12 + 0.005*(ldl - 120) - 0.003*(hdl - 50) + 0.002*(vldl - 30) + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    'LDL': ldl,
    'HDL': hdl,
    'VLDL': vldl,
    'TotalChol': total_chol,
    'Hemoglobin': hb
})

# 1) Таблиця кореляцій (Пірсон)
pearson_corr = df.corr(method='pearson')['Hemoglobin'].drop('Hemoglobin')
spearman_corr = df.corr(method='spearman')['Hemoglobin'].drop('Hemoglobin')

print("=== Коефіцієнти кореляції (Пірсон) з Hemoglobin ===")
print(pearson_corr.round(4))
print("\n=== Коефіцієнти кореляції (Спірмен) з Hemoglobin ===")
print(spearman_corr.round(4))

# 2) Статистична значущість (pearsonr + spearmanr)
print("\n=== Статистична перевірка (Pearson p-value) ===")
for col in ['LDL','HDL','VLDL','TotalChol']:
    r, p = pearsonr(df[col], df['Hemoglobin'])
    print(f"{col}: r={r:.4f}, p={p:.4e}")

print("\n=== Статистична перевірка (Spearman p-value) ===")
for col in ['LDL','HDL','VLDL','TotalChol']:
    r, p = spearmanr(df[col], df['Hemoglobin'])
    print(f"{col}: rho={r:.4f}, p={p:.4e}")

# 3) Візуалізація розсіяння та регресійних ліній
plt.figure(figsize=(10,8))
for i, col in enumerate(['LDL','HDL','VLDL','TotalChol']):
    plt.subplot(2,2,i+1)
    sns.regplot(x=col, y='Hemoglobin', data=df, scatter_kws={'s':20}, line_kws={'color':'red'})
    plt.title(f"Hemoglobin vs {col}")
plt.tight_layout()
plt.savefig("lipoprotein_hb_scatter.png")
print("\nГрафіки збережено у 'lipoprotein_hb_scatter.png'")