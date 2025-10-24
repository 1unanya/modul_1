import random

population = [{'id': i, 'fitness': random.uniform(0, 1)} for i in range(10)]

def tournament_selection(pop, k, t_size=3):
    winners = []
    for _ in range(k):
        group = random.sample(pop, t_size)
        winners.append(max(group, key=lambda x: x['fitness']))
    return winners

print("🔹 Початкова популяція:", population)
print("🔹 Відібрані (турнір):", tournament_selection(population, 3))