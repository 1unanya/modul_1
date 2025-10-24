import numpy as np

# Мамдані: нечітке правило IF–THEN
def mamdani_rule(x):
    if x > 0.7:
        return "Висока активність"
    elif x > 0.3:
        return "Середня активність"
    else:
        return "Низька активність"

for x in np.linspace(0, 1, 5):
    print(f"x={x:.2f} → {mamdani_rule(x)}")