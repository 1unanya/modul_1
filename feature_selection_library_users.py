import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Імітація даних активності користувачів
data = pd.DataFrame({
    'views': [15, 30, 5, 12, 50, 45],
    'downloads': [1, 5, 0, 1, 6, 5],
    'avg_read_time': [5, 12, 3, 4, 15, 13],
    'searches': [10, 25, 4, 9, 30, 28],
    'success': [0, 1, 0, 0, 1, 1]  # успішність користувача
})

X = data[['views', 'downloads', 'avg_read_time', 'searches']]
y = data['success']

selector = SelectKBest(score_func=f_classif, k=2)
selector.fit(X, y)
selected_features = X.columns[selector.get_support()]
print("Відібрані інформативні ознаки:", list(selected_features))