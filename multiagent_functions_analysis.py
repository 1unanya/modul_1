class Agent:
    def __init__(self, name):
        self.name = name

    def share_info(self, info, other):
        print(f"{self.name} ділиться '{info}' з {other.name}")

agent_a = Agent("Аналітик активності")
agent_b = Agent("Рекомендатор")

agent_a.share_info("користувач читає технічну літературу", agent_b)