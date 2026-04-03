import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from engine import DQNAgent

def run_sim(use_shaping):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(4, 2)
    scores = []
    for e in range(200):
        state, _ = env.reset()
        score = 0
        for _ in range(500):
            action = agent.select_action(state)
            ns, r, term, trunc, _ = env.step(action)
            # Shaping logic
            final_r = r - (0.01 * abs(ns[0])) - (0.1 * abs(ns[2])) if use_shaping else r
            agent.memory.push(state, action, final_r, ns, term or trunc)
            state, score = ns, score + r
            agent.update(64)
            if term or trunc: break
        if e % 10 == 0: agent.update_target()
        scores.append(score)
    return scores

sparse = run_sim(False)
shaped = run_sim(True)
plt.figure(figsize=(10,5))
plt.plot(sparse, label="Sparse")
plt.plot(shaped, label="Shaped")
plt.legend()
plt.title("Reward Shaping Comparison")
plt.savefig("results/reward_shaping.png")
print("Experiment 2 Complete: Plot saved.")
