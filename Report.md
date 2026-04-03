# Technical Report: DQN Stability in Nonstationary Environments

## 1. Abstract
This report evaluates the resilience of Deep Q-Learning (DQN) agents when subjected to environmental perturbations. We investigate the efficacy of target-network stabilization and experience replay in mitigating performance decay during distribution shifts, reward misspecification, and environmental nonstationarity.

## 2. Methodology
The agent utilizes a Deep Q-Network to approximate the action-value function $Q(s, a)$. The optimization target is defined by the Bellman equation:

$$Y_t = r_t + \gamma \max_{a'} Q_{target}(s_{t+1}, a'; \theta^{-})$$

Stability is maintained via a target network $\theta^{-}$ updated periodically from the policy network $\theta$. This decoupling prevents the "moving target" problem common in bootstrap-based learning.



## 3. Experiment 1: Distribution Shift via Mass Perturbation
* **Objective**: Measure value estimate degradation under physical perturbation.
* **Procedure**: Mid-training (Episode 200), the pole mass distribution was increased by a factor of 5 (length increased from **0.5** to **2.5**).
* **Results**: Observed an immediate policy collapse with a 90% reward drop.
* **Analysis**: Despite the shift, the agent successfully leveraged the **Experience Replay** buffer to re-map the modified state-action space, returning to baseline stability (~300+ score) within 100 episodes.

## 4. Experiment 2: Reward Shaping and Misspecification
* **Objective**: Contrast sparse reward signals $r_t \in \{0, 1\}$ with shaped signals $r_{shaped} = r_t + \Phi(s)$.
* **Procedure**: A potential-based penalty was added for pole angle $\theta$ and cart position $x$:
    
$$r_{shaped} = r_{base} - 0.01|x| - 0.1|\theta|$$

* **Findings**: Shaping accelerated early convergence (Episodes 0–75).
* **Policy Collapse**: In late-stage training, the agent began over-optimizing for penalty avoidance rather than the terminal goal, settling for a sub-optimal policy compared to the sparse baseline.

## 5. Experiment 3: Environmental Nonstationarity
* **Objective**: Observe convergence limits when the environment is a function of time (opponent behavior simulation).
* **Procedure**: Gravity was increased linearly: $g_t = 9.8 + (episode/10)$.
* **Mathematical Insight**: As $g \to \infty$, the state-transition probability $P(s'|s, a)$ shifts faster than the update frequency of the Replay Buffer.
* **Conclusion**: The agent maintained stability until $g \approx 17.0 \text{ m/s}^2$. Beyond this point, the "stale experience" in the buffer led to gradual performance decay.

## 6. Final Conclusions
The results confirm that while **Target Networks** and **Experience Replay** provide localized robustness, continuous nonstationarity requires adaptive buffer turnover to maintain convergence. This project builds the intuition necessary for deploying RL-based systems in unpredictable, real-world environments.
