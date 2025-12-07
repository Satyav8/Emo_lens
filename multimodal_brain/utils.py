from collections import deque

class EmotionHistory:
    def __init__(self, maxlen=10):
        self.valences = deque(maxlen=maxlen)
        self.arousals = deque(maxlen=maxlen)

    def add(self, v, a):
        self.valences.append(v)
        self.arousals.append(a)

    def momentum(self):
        if len(self.valences) < 2:
            return 0
        return (self.valences[-1] - self.valences[0]) / len(self.valences)

    def arousal_trend(self):
        if len(self.arousals) < 2:
            return 0
        return (self.arousals[-1] - self.arousals[0]) / len(self.arousals)
