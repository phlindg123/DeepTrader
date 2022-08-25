
import tensorflow as tf
import tensorflow.keras as k
from collections import deque
from Models import OU, ActorNetwork, CriticNetwork
import random
import numpy as np

class DeepTraderAgent:
    def __init__(self, state_size, action_dim, env, exploration="OU"):
        self.state_size = state_size
        self.action_dim = action_dim
        self.exporation = exploration
        self.env = env

        tau = 0.001
        sess = tf.Session()
        self.actor = ActorNetwork(self.state_size, self.action_dim, tau, sess)
        self.critic = CriticNetwork(self.state_size, self.action_dim, tau, sess)

        self.gamma = 0.99
        self.epsilon = 1
        self.loss = 0
        self.memory = deque(maxlen=5000)
        self.noice = OU(0.15, 0.2)
    def reset(self):
        self.loss = 0
    def act(self, state, p=0):
        if self.exporation == "OU":
            action = self.actor.model.predict(state)[0]
            for i in range(len(action)):
                action[i] = action[i] + (1-p)*next(self.noice)
            return action
        elif self.exporation == "E":
            if np.random.rand() <= self.epsilon:
                action = self.env.action_space.sample()
            else:
                action=self.actor.model.predict(state)[0]
            return action
       

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.asarray([e[0] for e in minibatch]).reshape((batch_size, self.state_size[0]))
        actions = np.asarray([e[1] for e in minibatch])
        rewards = np.asarray([e[2] for e in minibatch])
        next_states = np.asarray([e[3] for e in minibatch]).reshape((batch_size,self.state_size[0]))
        dones = np.asarray([e[4] for e in minibatch])
        y_t = np.asarray([e[2] for e in minibatch])
        

        target_actions =  self.actor.target_model.predict(next_states)
        target_q_values = self.critic.target_model.predict([next_states, target_actions])
        

        for k in range(batch_size):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.gamma*target_q_values[k]
        self.loss += self.critic.model.train_on_batch([states, actions], y_t)
        actions_for_grad = self.actor.model.predict(states)
        grads = np.array(self.critic.gradients(states, actions_for_grad))
        #print(grads.shape)
        self.actor.train(states, grads.reshape((batch_size,self.action_dim[0])))
        self.actor.train_target()
        self.critic.train_target()

        if self.exporation == "E":
            self.epsilon *= 0.995
    
    

    

    
        

        


    