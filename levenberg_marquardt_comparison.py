import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def model(x, a, b, c):
    return a * np.exp(-b * x) + c

x = np.linspace(0, 4, 50)
y = model(x, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x))

params, _ = curve_fit(model, x, y, method='lm')
plt.scatter(x, y, label='дані')
plt.plot(x, model(x, *params), color='red', label='LM fit')
plt.legend()
plt.show()