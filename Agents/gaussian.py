
import tensorflow as tf
import tensorflow.keras as k
from collections import deque
import numpy as np
from Models import VNetwork


"""Notes:
    w är alla vilkterna i critic networket

"""

class GaussianAgent:
    def __init__(self, state_size, action_dim, env):
        self.state_size = state_size
        self.action_dim = action_dim
        self.env = env
        
        self.sess = tf.Session()
        self.alpha_w = 1e-3
        self.alpha_theta = 1e-3
        self.lambda_w = 0.5
        self.lambda_theta = 0.5
        self.gamma = 0.99
        self.mu = 0
        self.sigma = 1
        self.theta_mu = np.zeros(self.action_dim)
        self.theta_sigma = np.zeros(self.action_dim)
        self.theta = np.array([self.theta_mu, self.theta_sigma])

        self.V = VNetwork(self.state_size, self.action_dim, self.alpha_w,self.sess)
        self.v_network, self.w = self.V.create_network()
        self.memory = deque(maxlen = 2000)

        self.v_grads = tf.gradients(self.v_network.output, self.w)

        self.sess.run(tf.global_variables_initializer())

    def act(self, state):
        #action = np.random.normal(self.mu, self.sigma, self.action_dim)
        action = np.random.uniform(size=self.action_dim)
        action = np.random.dirichlet(action.squeeze())

        return action
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
    def reset(self):
        self.I = 1
        self.loss = 0
    def _update_distr(self, portfolio, delta):
        portfolio = portfolio.reshape(self.action_dim)
        self.mu = np.dot(portfolio.T, self.theta_mu)
        self.sigma = np.exp(np.dot(portfolio.T, self.theta_sigma))
    #def _update_v(self, delta):
        
    def replay(self, batch_size):
        #Här kör vi nu bara en per tid, tror det är mer logiskt i finans perspektiv.
        state, action, reward, next_state, done = self.memory[-1]
        prices = state["prices"].reshape((batch_size, self.state_size[0], self.state_size[1]))
        portfolio = state["portfolio"].reshape((batch_size, self.action_dim[0], self.action_dim[1]))
        next_prices = next_state["prices"].reshape((batch_size, self.state_size[0], self.state_size[1]))
        next_portfolio = next_state["portfolio"].reshape((batch_size, self.action_dim[0], self.action_dim[1]))
        delta = 0
        target_v = self.v_network.predict([next_prices, next_portfolio])
        v = self.v_network.predict([prices, portfolio])
        if done:
            delta = reward - v
        else:
            delta = reward + self.gamma*target_v - v
        
        self.loss += self.v_network.train_on_batch([prices, portfolio], delta.reshape(1,1))
        self._update_distr(portfolio, delta)
        self._update_v(delta)
    def run(self, episodes):
        self.losses = []
        for episode in range(episodes):
            state = self.env.reset()
            self.reset()
            done = False
            tot_reward = 0
            while not done:
                action = self.act(state)
                #print(action)
                next_state, reward, done, _ = self.env.step(action)
                tot_reward += reward
                self.remember(state, action, reward, next_state, done)
                self.replay(1)
                state = next_state
            print("Episode {}/{}, Total reward: {}, Loss: {}".format(episode+1, episodes, tot_reward, self.loss))
            self.losses.append(self.loss)
            
        

    
    