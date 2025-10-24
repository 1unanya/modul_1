class Agent:
    def __init__(self, name):
        self.name = name
        self.peers = []

    def connect(self, other):
        self.peers.append(other)
        other.peers.append(self)

    def send(self, msg):
        for peer in self.peers:
            print(f"{self.name} → {peer.name}: {msg}")

a1 = Agent("Агент моніторингу")
a2 = Agent("Агент аналітики")
a3 = Agent("Агент рекомендацій")

a1.connect(a2)
a2.connect(a3)

a1.send("користувач знизив активність")