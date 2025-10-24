import numpy as np

# Матриця відстаней між “сторінками” бібліотеки
distances = np.array([[0, 2, 2, 5],
                      [2, 0, 3, 2],
                      [2, 3, 0, 3],
                      [5, 2, 3, 0]])

pheromone = np.ones_like(distances)
alpha, beta = 1, 2

# Розрахунок ймовірності вибору маршруту
visibility = 1 / (distances + 1e-10)
prob = (pheromone ** alpha) * (visibility ** beta)
prob /= prob.sum(axis=1, keepdims=True)
print("🔹 Ймовірності переходів (ACO):\n", prob)