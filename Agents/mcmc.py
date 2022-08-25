

import numpy as np
from collections import deque



class MCAgent:
    def __init__(self, state_size, action_dim, env, exploration="E"):
        self.state_size = state_size
        self.action_dim = action_dim
        self.env = env
        self.exploration = exploration

        self.memory = deque(maxlen=2000)

        self.q_table = np.zeros((state_size, action_dim))
        self.N = 100
        self.actions = [1/self.N for i in range(self.N)]
        self.weights = [1/self.N for i in range(self.N)]
        self.pi = self._create_action_distr()


    def _create_action_distr(self):
        pi = []
        def dirac(x, x0):
            if all(x-x0 == 0):
                return 1
            else:
                return 0
        for i in range(self.N):
            pi_i = 0
            for j in range(self.N):
                pi_i += self.weights[j]*dirac(self.actions[i], self.actions[j])
            pi.append(pi_i)
        return pi
    
    def act(self, state, p):
        if np.random.rand() <= p:
            action = self.env.action_space.sample()
        else:
            action = np.random.choice(self.actions, p=self.pi)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def update_critic(self):
        state, action, reward, next_state, done = self.memory[-1]
        

    def replay(self, batch_size):


