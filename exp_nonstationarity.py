import gymnasium as gym
import matplotlib.pyplot as plt
from engine import DQNAgent

env = gym.make("CartPole-v1")
agent = DQNAgent(4, 2)
scores = []

for e in range(200):
    env.unwrapped.gravity = 9.8 + (e / 10.0) # Escalating gravity
    state, _ = env.reset()
    score = 0
    for _ in range(500):
        action = agent.select_action(state)
        ns, r, term, trunc, _ = env.step(action)
        agent.memory.push(state, action, r, ns, term or trunc)
        state, score = ns, score + r
        agent.update(64)
        if term or trunc: break
    if e % 10 == 0: agent.update_target()
    scores.append(score)

plt.figure(figsize=(10,5))
plt.plot(scores)
plt.title("Nonstationary Gravity Performance")
plt.savefig("results/nonstationarity.png")
print("Experiment 3 Complete: Plot saved.")
