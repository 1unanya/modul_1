import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Синтетичні дані: ознаки користувацької активності у бібліотеці
np.random.seed(42)
n = 500
views = np.random.poisson(lam=20, size=n)
downloads = np.random.poisson(lam=2, size=n)
avg_read_time = np.random.normal(loc=8, scale=4, size=n).clip(0.5)
searches = np.random.poisson(lam=5, size=n)
video_watch_ratio = np.random.beta(a=2, b=5, size=n)

# Цільова змінна формуємо як функція ознак з шумом
score = 0.03*views + 0.2*downloads + 0.1*avg_read_time + 0.05*searches + 1.5*video_watch_ratio
prob = 1 / (1 + np.exp(- (score - 2.5)))  # сигмоїда
success = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    'views': views,
    'downloads': downloads,
    'avg_read_time': avg_read_time,
    'searches': searches,
    'video_watch_ratio': video_watch_ratio,
    'success': success
})

# 2) Поділ на train/test
X = df[['views', 'downloads', 'avg_read_time', 'searches', 'video_watch_ratio']]
y = df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# 3) Навчання дерева рішень (прості налаштування)
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0)
clf.fit(X_train, y_train)

# 4) Оцінка моделі
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== Оцінка дерева рішень ===")
print("Accuracy:", round(acc, 4))
print(report)
print("Confusion matrix:\n", cm)

# 5) Вивід структури дерева у текстовому вигляді
print("\n=== Структура дерева (export_text) ===")
rules_text = export_text(clf, feature_names=list(X.columns))
print(rules_text)

# 6) Візуалізація дерева (збереження у файл)
plt.figure(figsize=(14,8))
plot_tree(clf, feature_names=X.columns, class_names=['not_success','success'], filled=True, rounded=True)
plt.title("Decision Tree for Library User Engagement")
plt.tight_layout()
plt.savefig("decision_tree_plot.png", dpi=150)
print("\nДерево збережено як 'decision_tree_plot.png'")

# 7) Додатковий аналіз важливості ознак
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Важливість ознак ===")
print(importances)

# Показати графік важливостей
plt.figure(figsize=(6,4))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("Графік важливостей збережено як 'feature_importances.png'")