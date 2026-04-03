import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from engine import DQNAgent

env = gym.make("CartPole-v1")
agent = DQNAgent(4, 2)
scores = []

for e in range(400):
    if e == 200: env.unwrapped.length = 2.5 # Shift mass
    state, _ = env.reset()
    score = 0
    for _ in range(500):
        action = agent.select_action(state)
        ns, r, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.memory.push(state, action, r, ns, done)
        state, score = ns, score + r
        agent.update(64)
        if done: break
    if e % 10 == 0: agent.update_target()
    scores.append(score)

plt.figure(figsize=(10,5))
plt.plot(scores, alpha=0.6)
plt.axvline(x=200, color='r', linestyle='--', label='Shift')
plt.title("Distribution Shift Performance")
plt.savefig("results/distribution_shift.png")
print("Experiment 1 Complete: Plot saved.")
