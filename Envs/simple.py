import pandas as pd
import numpy as np
import gym

class Ticker:
    def __init__(self, name, length):
        self.name = name
        self.length = length
        
        self.pnl = 0
        self.s0 = 100#250 * np.random.rand()
        self.t = 0
        self.mu = np.random.normal(loc=0.05, scale=0.1)
        self.sigma =1.0 / np.random.gamma(shape=3, scale=8)
        self.st = self.s0
        self.weight = 0
        
    def __next__(self):
        if self.t >= self.length:
            raise StopIteration
        val = self.st
        self.t += 1
        B = np.random.randn()
        dt = 1 / self.length
        self.st = self.st * np.exp( (self.mu - 0.5 * self.sigma**2)*dt + self.sigma*np.sqrt(dt)*B)
        self.pnl = self.st - val
        return self.st
    
    def __iter__(self):
        return self
    
    def reset(self):
        self.t = 0
            

class Simple:
    def __init__(self, length):
        self.length = length        
        self.ticker = Ticker("Ticker", length)
        
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float)

    def reset(self):
        self.signals = [0]
        self.t = 0
        self.weights = [0]
        self.cash = 1000.0
        self.pnl = [0]
        self.past_prices = [next(self.ticker)]
        self.ticker.reset()
        return np.array([0,0])
    
    def _calculate_signals(self):
        if self.t > 30:
            long_avg = np.mean(self.past_prices[-30:])
            short_avg = np.mean(self.past_prices[-10:])
            return short_avg / long_avg -1
        else:
            return 0
        
    def step(self, action):
        self.t += 1
        w = self.weights[-1] + action
        
        new_price = next(self.ticker)
        self.past_prices.append(new_price)
        self.pnl.append(w * self.ticker.pnl)
        self.cash += w*self.ticker.pnl
        self.weights.append(w)
        signal = self._calculate_signals()
        self.signals.append(signal)
        
        done = self.t >= self.length-1
        state = [w, signal]
        reward = self.pnl[-1]
        
        return np.array(state), reward, done, None
        
        
        
        