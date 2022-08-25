
import gym
import pandas as pd
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, data, window_length=50):
        self.data = data.values #ska ha shape (n_assets, antal_steg, antal_features = 1 oftast. HAR INTE MED NU)
        self.rets = data.pct_change().fillna(0).values
        print(self.rets.shape)

        self.time = data.index
        self.n_assets = self.data.shape[0]

        self.window_length = window_length
        #self.init_invest = init_invest

        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets+1,) #cash ska också ha en weight!
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets+1,)
        )

        self._step = self.window_length
        self.max_step = self.data.shape[1]

        self.current_weights = [1] + [0]*self.n_assets
        self.p0 = 1
        self.infos = []

        #Hyperparamters
        self.risk_aversion = 500
        

    def calculate_reward(self, weights):
        assert self._step >= self.window_length
        curr_rets = self.rets[:, (self._step-self.window_length):self._step+1]
        
        #cov = np.cov(curr_rets)
        asset_weights = weights[1:]
        #vol_loss = self.risk_aversion * np.transpose((self.current_weights[1:] + asset_weights)) @ cov @ (self.current_weights[1:] + asset_weights)
        pure_ret = np.dot(curr_rets[:,-1], asset_weights)
        return pure_ret#, vol_loss
    def sharpe(self):
        rets = [x["cur_ret"] for x in self.infos]
        return np.mean(rets)/np.std(rets)
    def step(self, action):
        info = {}
        weights = action/np.sum(action)
        #print(weights)
        assert self.action_space.contains(weights)
        pure_ret = self.calculate_reward(weights)
        self.current_weights += weights
        self.current_weights /= np.sum(self.current_weights)
        self.p1 = self.p0 *(1.0 + pure_ret)
        
        info["cur_ret"] = self.p1 - self.p0
        info["cur_val"] = self.p1
        self.infos.append(info)
        self._step += self.window_length
        done = self._step >= self.max_step or self.p1==0
        if self._step >= self.max_step - self.window_length:
            s = self.sharpe()
            print("CALCULATING SHARPE")
            if s < 0:
                reward = -1000
            elif s > 0 and s < 1:
                reward = -100
            elif s > 1 and s < 2:
                reward = 100
            elif s > 2:
                reward = 1000
        else:
            reward = pure_ret
        
        
        
        return np.array(self.current_weights), reward, done, info
    def reset(self):
        self._step = self.window_length
        self.current_weights = [1] + [0]*self.n_assets
        self.p0 = 1
        self.infos = []
        return np.array(self.current_weights)
        
        





    
    







