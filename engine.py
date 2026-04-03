import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x): return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), action, reward, np.array(next_state), done)
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma, self.epsilon = gamma, 1.0
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad(): return self.policy_net(state).argmax().item()

    def update(self, batch_size):
        if len(self.memory) < batch_size: return
        s, a, r, ns, d = self.memory.sample(batch_size)
        s, ns = torch.FloatTensor(s).to(device), torch.FloatTensor(ns).to(device)
        a, r, d = torch.LongTensor(a).to(device), torch.FloatTensor(r).to(device), torch.FloatTensor(d).to(device)
        
        curr_q = self.policy_net(s).gather(1, a.unsqueeze(1))
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0]
            target_q = r + (self.gamma * max_next_q * (1 - d))
        
        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
